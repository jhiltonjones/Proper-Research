# K2 revised procedure — draft for review (2026-11-14)

**Status: DRAFT. Nothing below is decided or in effect. Nothing in this document retroactively
changes or replaces the two existing K2 results.**

## The two existing K2 runs remain the official prior record

- N=15: `close_loop_logs/myrun/k2_timing_check_20261111T181150Z_20261111T181208Z/` — official
  5-fold result `[2,2,2,1,2]`, G_2 never exceeding 0.9 on any fold → **STOP, d=2 not confirmed.**
- N=3: `close_loop_logs/myrun/k2_timing_check_20261111T181443Z_20261111T181501Z/` — official
  5-fold result `[1,1,1,1,1]` → **STOP, d=2 not confirmed**, and not comparable to N=15 (different
  horizon condition).

Both STOP conclusions stand. This document describes what the *next* data collection should look
like so the next attempt can actually distinguish the open questions the first two runs left behind —
it is not a retroactive reinterpretation of what already happened.

## What the new instrumentation lets you distinguish that the old logs couldn't

The old `path_follow.jsonl` logged only `target_update_ms` (the lock-acquisition time for handing a
target to the streaming thread — microseconds, never the real bottleneck) and nothing at all about
the streaming thread's own behavior. The new `arm_streamer_events.jsonl` (per-cycle, written by
`BackgroundArmStreamer`'s new event log — see `close_loop_path_follow.py`) adds, for the first time:

1. **Command-pickup delay** — the real gap between a target being published (`update_target()`) and
   the streaming thread actually reading it (`pickup` event, `pickup_latency_s` field). The old logs
   could not distinguish "the target was handed off instantly" from "the target sat unread for up to
   one 20ms cycle" — now you can see the real distribution.
2. **State-measurement age at publish time** — each `publish` event now carries the `state_age_s` the
   main loop had at the moment it computed that target, so you can correlate a stale measurement with
   whatever happens downstream.
3. **Real physical joint response vs. commanded** — this was already partially possible via
   `q_meas_rad`/`q_target_delta_rad` in `path_follow.jsonl` (100ms granularity); the new
   `servo_j_call` events (start/end/duration, realized interval) let you see whether the 20ms-paced
   streaming layer itself is keeping up, which `path_follow.jsonl` alone could never show (confirmed
   by the mocked test in `test_close_loop_path_follow_streamer.py`: a slowed-down `servo_j()` now
   visibly drops the realized rate in the log, which the old design would have silently missed).

## Proposed changes for the next K2 run — each needs your explicit agreement, none is decided

- **Proposed: use the consolidated `k2_delay_and_step_analysis.py` script** (new,
  `proper_research/studies/mpc_ablations/k2_delay_and_step_analysis.py`) instead of ad-hoc
  calculations, so every future delay/step-realization number is computed with the same documented,
  named parameters (tick pairing, epsilon, row selection) as this round's reconciliation, and pinned
  to the exact input file hashes it read.
  - *Rationale:* this round's investigation found the step-realization medians are sensitive to an
    undocumented epsilon choice (eps=0 reproduces the historical numbers; eps=1e-5 shifts several
    joints' medians by 0.02-0.03). Making that choice explicit and versioned avoids silently
    re-deriving a different number next time.
- **Proposed: increase per-fold excitation, or stratify the 5-fold split by excitation rather than by
  contiguous time blocks.**
  - *Rationale:* this round found per-fold commanded-delta RMS varies ~3x across the N=15 run's 5
    folds, and the identification quality (`G_2`) does not cleanly track which fold has the least
    excitation — a stratified or excitation-matched split might give a cleaner signal, but this is a
    real change to the identification *methodology*, not something to adopt without your sign-off.
- **Proposed: a minimum-commanded-magnitude floor before including a tick in the step-realization
  ratio**, documented and versioned (e.g. the `--step-eps` flag in the new script), rather than an
  implicit zero-floor.
  - *Rationale:* this round's tercile analysis showed ratio noise exploding at near-zero commanded
    magnitudes (e.g. joint 5's lowest tercile: std=30.96, range up to 252 — pure division-by-near-zero
    noise, not a real actuator-gain signal). Whether to floor this, and at what value, is a protocol
    decision, not something this analysis should decide unilaterally.
- **Proposed: a longer or more-excited run** (more ticks, or a path with less low-command dead time)
  specifically so the 5-fold check has fewer near-zero-excitation ticks diluting each fold.

**Nothing above is applied to the stop rule itself.** The `G_2 > 0.9` / `>=4/5 folds` criterion is
unchanged; these are proposals about how to *collect and process* the next dataset the rule gets
applied to, not about loosening the rule.

## Next safe step

Offline only, until you've reviewed and either accepted or revised the proposals above:
run the new `k2_delay_and_step_analysis.py` against the two existing logs (already done this round,
reproduced the official numbers exactly) so it's validated against known-good data before relying on
it for a new run; review the new `arm_streamer_events.jsonl` schema and confirm it captures what you
need; decide on the excitation/epsilon proposals above. No robot motion is implied or required by
anything in this document.

---

## Event/field schema (2026-11-15 addendum) — five easily-conflated timing quantities, explicitly distinguished

| Field (as logged) | File | Source (file:line) | What it measures | Why it's NOT its neighbor |
|---|---|---|---|---|
| `state_age_s` | `path_follow.jsonl` (existing) + `arm_streamer_events.jsonl` `publish` event (new) | `close_loop_path_follow.py:1959` (`estimate, age = camera.latest(cfg.max_state_age_s)`) → `camera_source.py:249` (`CameraSource.latest()`) | **Camera/vision age**: how stale the vision-derived tip estimate is when the main loop uses it to compute this tick's command. | Comes from the camera/vision pipeline specifically — a different physical sensor with its own frame-capture and processing latency, not the robot's joint encoders. |
| *(not currently logged — identified gap)* | — | `state_stream.py:285-287` (`RobotJointStream.latest_joints()`: `value, _age = self.joints_slot.get(max_age_s); return value`) | **Joint-sample age**: how stale the measured joint state is. The age IS computed internally by `LatestSlot.get()` but is discarded before `latest_joints()` returns it. | Comes from a *separate* RTDE polling thread (`state_stream.py:151`, its own `poll_hz`), independent of the camera's frame rate — vision and joints can be stale by different amounts at the same instant. **This is a real, currently-unclosed gap**, parallel to the servo-duration gap this round already fixed; closing it would require `latest_joints()` to return (and the caller to log) the age it already computes and throws away. Not fixed in this pass — flagged for a future pass. |
| `target_update_ms` | `path_follow.jsonl` (existing) | `close_loop_path_follow.py:181-199` (`BackgroundArmStreamer.update_target()`) | **Handoff time**: the duration of acquiring the lock, copying the 6-vector, and (after this round's instrumentation) emitting the `publish` event — measured from the main control loop's side. | This is NOT pickup latency (the earlier, now-corrected mistake). It only measures how long the *publish-side* operation takes, never whether or when the streaming thread actually reads the value. |
| `pickup_latency_s` (in the new `pickup` event) | `arm_streamer_events.jsonl` (new) | `close_loop_path_follow.py:236-243` (`BackgroundArmStreamer._run()`, `pickup` event) | **Pickup time**: the real elapsed time from `update_target()`'s publish timestamp to the streaming thread's next cycle actually reading that target. | This is the thing `target_update_ms` was previously (wrongly) assumed to represent — it is measured on the *consumer* side (the streaming thread), not the publisher side, and can be up to one full streamer cycle (~20ms) even though `target_update_ms` itself is microseconds. |
| `servo_duration_ms` (in the new `servo_j_call` event) | `arm_streamer_events.jsonl` (new) | `close_loop_path_follow.py:265-269` (`BackgroundArmStreamer._run()`, wrapping `self._robot.servo_j(...)`) | **Servo-call duration**: the real wall-clock time the `servo_j()` RTDE call itself takes to return. | Previously computed as `last_servo_ms` and silently discarded (confirmed in the 2026-11-14 pass of this investigation) — never logged anywhere before this round. Distinct from pickup time: pickup is "how long before the thread *noticed* the target," this is "how long the actual RTDE call *took*" once it started. |

## Explicit statement for the record

> **This instrumented repeat, if and when the user runs it, collects DIAGNOSTIC EVIDENCE ONLY. It does
> not itself constitute Day-1 clearance.** Day-1 clearance requires the existing K2 go/no-go criterion
> — the 5-fold delay check confirming d=2 (≥4/5 folds, G_2 > 0.9 on every fold) — to actually be
> satisfied. Neither of the two existing K2 runs (N=15 or N=3) met that criterion; both remain
> official STOP results. Running the instrumented repeat below, even if it completes cleanly and even
> if its new timing fields look good, is **not** equivalent to passing K2, and must not be read or
> reported as such.

---

## Source-diff review (2026-11-15): is the instrumentation control-behavior-neutral?

**Mostly yes, with one honest exception that matters for interpreting the first instrumented run.**

Reviewed line-by-line (`BackgroundArmStreamer`, `close_loop_path_follow.py`): the `servo_j()` call's
arguments (`target`, `time_s`, `lookahead_time`, `gain`) are byte-identical before and after. The
scheduler logic (`next_cycle += self._stream_interval_s`, the sleep-until-deadline pattern) is
unchanged. No existing variable's computed value changed. The new code is additive: new fields on
`__init__`, a new `enable_event_log()`/`_emit()`/`_writer_loop()` trio, and new (non-blocking
`queue.put_nowait`) emit calls inserted at four points.

**The one real exception:** two of those new emit calls sit in the hot path *before* a
previously-existing timing measurement completes:
1. In `update_target()`, the new `publish` emit happens *before* the function returns its elapsed-ms
   value — so `target_update_ms` in the new instrumented log now includes a small (dict-construction +
   `queue.put_nowait`) overhead it did not include before. This metric is not measuring the exact same
   thing it measured in the original K2 runs.
2. In `_run()`, the new `pickup` emit happens immediately before `t0 = now_monotonic()` and the actual
   `servo_j()` call — adding the same small, non-blocking-but-nonzero overhead immediately before each
   servo call.

Both additions are genuinely tiny (a dict + a non-blocking queue operation, no I/O, no lock contention
with the writer thread) — almost certainly sub-100-microsecond, utterly negligible against the
20-100ms timescales that matter for deadline misses or period. But it means **the new instrumented
run cannot be treated as a bit-for-bit identical replay of the old runs' real timing** — it is new
diagnostic data about a very slightly different (by microseconds) execution, not a perfect
apples-to-apples re-run. This is exactly why the statement below exists.

**Mock checks confirmed against current code:** slow-servo_j test (40ms call → 25.0Hz realized vs.
50.0Hz configured) still passes; new orderly-shutdown test (50 publishes issued in a tight burst, all
50 found in the log after `stop()` returns) passes. Both run for real, output shown in the main report.

## Exact command for one instrumented N=15 diagnostic repeat (prepared, NOT run)

Pulled directly from the original N=15 run's own recorded `argv` in `controller_metadata.json` —
every flag matches exactly except `--run-name`:

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

Expected output files in the new run directory (verify all are present after running):
- `controller_metadata.json` — now with `code_provenance` (commit hash + dirty flag), `plan_file_sha256`,
  `schedule_cache_sha256`, `schedule_cache_path` added.
- `summary.json`, `path_follow.jsonl`, `path_follow_plot.png`, `predicted_beam_positions.jsonl`,
  `startup_stationary_tips.jsonl`, `tip_trajectory.csv`, `frozen_jacobian.json` — same as before.
- **`arm_streamer_events.jsonl` — new.** One line per streamer-thread event (`publish`/`pickup`/
  `servo_j_call`/`idle_no_target`), per the schema above.

**This repeat, if run, is diagnostic evidence only — see the explicit statement above. It is not a K2 pass.**

---

## 2026-11-15 correction: authoritative baseline, exact formula (no more convention-searching)

**The prior "reconciled" baseline above used a grid search to pick whichever tick-pairing/epsilon
combination minimized error against old reported numbers. That method was wrong and has been
removed from `k2_delay_and_step_analysis.py` entirely.** The script now implements exactly one
formula, matching `HOWTO_CAMPAIGN2.md`'s own prior documentation (same-tick pairing, denominator
`np.maximum(np.abs(commanded), 1e-12)`, all rows, no masking) — confirmed by direct citation of that
file (lines ~368-382), not by searching for a match.

Run against both SHA-256-pinned logs with this exact formula:

| Run | Step-realization medians (exact formula) | vs. historically reported |
|---|---|---|
| N=15 | `[0.918, 1.026, 0.919, 0.874, 0.843, 1.020]` | **Exact match**, every value |
| N=3 | `[0.685, 0.996, 0.772, 0.706, 0.547, 0.887]` | No prior number existed; this is now the authoritative N=3 baseline |

The earlier near-miss (medians off by 0.01-0.03) came from an incorrect implementation (masking out
near-zero-command ticks instead of clamping the denominator per the actual spec) — not from the
historical numbers themselves being wrong. No number is being "corrected away" here; the exact
documented formula reproduces the N=15 numbers exactly.

## Age-field schema — joint-measurement gap now closed

- **`camera_vision_age_s`** (renamed/clarified from the pre-existing `state_age_s`, both now logged) —
  source: `close_loop_path_follow.py:1959` → `camera_source.py:249`. Unchanged from before.
- **`joint_sample_host_age_s`** (NEW, gap closed 2026-11-15) — source: `RobotJointStream.
  latest_joints_with_age()`, added at `state_stream.py` (new method alongside the unchanged
  `latest_joints()` — all 7 existing callers of `latest_joints()` are untouched). Logged into
  `path_follow.jsonl` at the call site `close_loop_path_follow.py` (inserted right after the existing
  `q = np.asarray(estimate.robot_joints, ...)` line). This is the HOST's receipt timestamp age (when
  this process's RTDE-polling thread wrote the value), not a robot-side timestamp.
- **`robot_acquisition_age_s`** — explicitly logged as `None` with an in-code comment stating why:
  `rtde_receive.RTDEReceiveInterface` has its own `getTimestamp()` in the real RTDE API, but this
  codebase's `RobotJointStream._poll_body()` never calls it (confirmed by reading that method
  directly — only `getActualQ()`/`getActualTCPPose()` are called). Not fabricated; genuinely not
  available from what this code currently captures. Adding it would need hardware to verify against,
  which this session doesn't have.

## Handoff-metric fix, measured

`target_update_ms`'s definition is restored exactly to its original meaning: the elapsed-time
measurement in `update_target()` is now computed BEFORE the event-log emit call (not after, as the
first instrumentation pass had it), so the logged value no longer includes any event-logging
overhead. Measured directly (2000 calls each, logging on vs. off): difference is
**-0.00006ms mean / 0.00001ms median** — within measurement noise, not a systematic inflation. The
fix is confirmed to work, not just asserted.

## Final diff scope (all passes, this whole investigation)

Full line-level diffs were generated against each file's true pre-investigation original (not shown
in full here — available on request) for: `close_loop_path_follow.py` (BackgroundArmStreamer
instrumentation + handoff fix + joint-age logging), `state_stream.py` (`latest_joints_with_age()`
addition), `common.py` (`capture_code_provenance()` addition), `run_mpc_delay_aware_insertion_anchor.py`
(provenance fields in `controller_metadata.json`), `k2_delay_and_step_analysis.py` (rewritten twice:
first with the wrong grid-search method, now with the exact documented formula),
`test_close_loop_path_follow_streamer.py` (new, now 3 passing checks: fast/slow servo_j rate,
orderly shutdown). All changes confirmed additive/corrective to timing-measurement and logging code
only — no change to `delay_samples`, the stop rule, or any control/command-generation logic anywhere
in this investigation.

## Instrumented N=15 repeat command — reconfirmed unchanged

Same command as before (settings/limits unchanged); expected output files now also include the three
new age fields inside `path_follow.jsonl` (no new file for these — they're additive fields on the
existing per-tick row) alongside `arm_streamer_events.jsonl`:

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

**Diagnostic evidence only — not a K2 pass. The stop rule and both existing official STOP results are
unchanged by any of this work.**

---

## 2026-11-16 correction: atomic joint-sample snapshot (supersedes the two-callback proposal)

**A real gap was found and is now closed.** The `joint_sample_host_age_s` field added on 2026-11-15
was computed from a SEPARATE, LATER call to the joint stream than the one that produced
`q_meas_rad` — architecturally two different reads that could, in principle, observe two different
samples (confirmed possible, not just theoretical, by tracing `estimate.robot_joints`'s real origin
into `CameraSource`'s background frame-processing thread, which runs asynchronously from the main
control loop).

**Implemented fix (not the originally-proposed two-callback version, which had the same atomicity
flaw one level up):**
- `LatestSlot.get_with_timestamp()` (`messages.py`) — one atomic (value, raw monotonic timestamp)
  read under a single lock acquisition.
- `RobotJointStream.latest_joints_with_timestamp()` (`state_stream.py`) — exposes that atomic read.
- `StateEstimate.robot_joints_stamp` (`messages.py`) — new field, paired with `robot_joints`, set from
  the SAME atomic read via a new opt-in `robot_joints_with_timestamp_getter` constructor parameter on
  `CameraSource` (`camera_source.py`) — mutually exclusive with the legacy `robot_joints_getter`, which
  is completely unchanged for its other 3 existing callers.
- `StateEstimate.robot_joints_age(ref=...)` — recomputes age from the stored stamp at any later moment,
  without re-reading the stream.
- The runner (`close_loop_path_follow.py`) now logs **`joint_sample_age_at_use_s`** (computed where
  `q_meas_rad` is actually read and used) and **`joint_sample_age_at_log_s`** (recomputed from the same
  stamp at row-construction time) — two distinct, separately-named fields, since real time passes
  between those two moments (tracking-error check, QP solve, etc.). `robot_acquisition_age_s` remains
  explicitly `None` — unchanged, still not fabricated.

**Measured overhead:** the new atomic read (`get_with_timestamp()`) is marginally *faster* than the
old `get()` (-0.17µs/call, within measurement noise either way) since it skips an internal age
subtraction. More importantly, this read happens inside `CameraSource`'s background frame-processing
thread, not the 100ms-tick control loop — it does **not** touch `target_update_ms` or anything in the
timing-critical path the earlier streamer instrumentation affected. **Not claiming bit-for-bit timing
equivalence with the pre-instrumentation runs overall** — the earlier, separately-measured
`target_update_ms` overhead (already fixed and quantified at ~0 after that fix) and this change are
different code paths; only `target_update_ms` has been directly, numerically confirmed unaffected.

**Regression test**, `test_atomic_joint_snapshot.py` (3 checks, run for real, all passed): `LatestSlot`
atomicity under a simulated concurrent update; `StateEstimate`'s joints/stamp pair staying consistent
even when the underlying slot advances after construction (with the OLD two-call pattern reproduced
explicitly to show the race it would have hit); age correctly growing from a held stamp across
multiple independent recomputations using a mocked clock.

## Plan/schedule identity for the diagnostic repeat (resolved)

The prepared N=15 diagnostic-repeat command's `--plan-dir`/`--schedule-cache` point to
`plans/rectangle_10x15mm_skipglobal_z4b_2026-11-11/...` — **identical path** to the original two K2
runs' own recorded `argv`. This is a **different** directory from
`plans/rectangle_10x15mm_skipglobal_z4b_marginfix_2026-11-11/` (the lower-acceleration-safety-factor
rebuild done earlier in this session for an unrelated live-arm QP-infeasibility problem, not part of
this K2 delay investigation) — the marginfix plan is not used anywhere in this diagnostic repeat.

Content, not just path, independently verified for the plan file: live SHA-256 of
`time_parameterized_configuration_path.npz` = `62cf493b0dc48518985e19fb09c97021de8434b7ca2a92a3a7a4229602453613`,
**exact match** to `protocol_campaign2.yaml`'s own `rectangle_mpc_sj_n15.plan_sha256` field. Note: this
confirms the file hasn't changed since the protocol was built — it does **not** confirm identity
against the original K2 runs' actual inputs directly, since neither original run's own
`controller_metadata.json` recorded a hash at all (they predate the provenance-repair work in this
investigation). For the schedule file (`jacobian_schedule_insertion_anchor.npy`), no historical hash
exists anywhere — not in the original runs, not in the protocol file — so its content-identity cannot
be independently verified beyond "the file at this path hasn't been touched by any edit in this
investigation."
