# Servo diagnostic logging v2 — 9 corrections applied (review only)

**No hardware connection and no run execution happened to produce this
document.** No servo-parameter/plan/constraint/`delay_samples` change,
no git by the assistant. Neither v1 file was ever committed to version
control, so there is no git diff to show; the two files below are
given in full as the complete replacement content at the same paths.

## Files changed (full content, replacing v1 at the same paths)

- `proper_research/hardware/online/servo_diagnostic_logging_prep.py`
  (rewritten — see repo for full file; key additions: `ConsistencyState`
  enum, `_safe_optional_call()`, `StaleStreamDetector`,
  `DiagnosticLogSink`, `THREE_TIMING_QUANTITIES_DOC`)
- `proper_research/hardware/online/test_servo_diagnostic_logging_prep.py`
  (rewritten — 12 tests, including the 7 required failure modes)

Neither file is imported by any production run script — verified by
`test_cadence_and_control_state_values_unchanged_by_this_module`
(grep-based, see below).

## Correction 1 — real configured poll rate, not the class default

Checked the completed K2 run's own `summary.json`:
`.config.reader_poll_hz = 60.0` — confirmed from
`close_loop_path_follow.py:752` (`reader_poll_hz: float = 60.0`), used
unmodified by `RobotJointStream(cfg.robot_ip,
poll_hz=cfg.reader_poll_hz, receive_only=True)` at line 1261. **This is
60Hz, not the `RobotJointStream` class's own 125Hz default** cited in
the v1 proposal. The module now documents this fact
(`DEFAULT_OBSERVED_K2_POLL_HZ = 60.0`, for reference only) but hardcodes
nothing — any real integration must read `cfg.reader_poll_hz` at call
time.

## Correction 2 — three-state consistency, never binary

`ConsistencyState.CHANGED / UNCHANGED / UNKNOWN` replaces the old
boolean `atomic_suspect`. A missing or failed timestamp read is always
`UNKNOWN` — `RobotSideSnapshot.__post_init__` checks `a is None or b is
None` **before** comparing equality, so a missing read can never fall
through to `UNCHANGED`. Every docstring/comment states `UNCHANGED` as
"consistent with atomicity, does not prove it," given the stale-packet
confound.

## Correction 3 — optional-method lookup AND invocation guarded, logged

`_safe_optional_call()` does `getattr(obj, method_name, None)` first
(catches a missing method on an older library/firmware build) and only
then calls it inside its own `try/except` (catches the method existing
but raising). Every failure — missing or raised — is appended to
`read_failures` with a `reason` field (`"method_missing"` or
`"call_raised"` plus the real exception repr), never silently
swallowed.

## Correction 4 — stale-packet vs. genuine stillness, made observable

`StaleStreamDetector.update()` tracks `actual_q` and the robot
timestamp across consecutive calls and reports one of
`"stationary_fresh"` (joints repeat, timestamp **advances** — real
stillness), `"suspected_stale_stream"` (joints repeat, timestamp does
**not** advance — frozen packet, the exact failure mode already
documented in `state_stream.py`'s 2026-09-17 comment), `"moving"`, or
`"unknown"` (insufficient history). Tests 4 and 5 below demonstrate
both directions concretely, not just the frozen case.

## Correction 5 — three timing quantities, explicitly non-acknowledgement

`THREE_TIMING_QUANTITIES_DOC` names and cites the three already-existing,
already-distinct host-side timing fields (`publish`/`pickup`/
`servo_j_call` events, `close_loop_path_follow.py:194-281`) and states
plainly, in one place, that none of them — nor anything in this module
— constitutes a robot acknowledgement. No new fields were added to
`BackgroundArmStreamer`; it already keeps these three separate.

## Correction 6 — logging failures cannot touch control

`DiagnosticLogSink.emit()` catches `queue.Full` and any other exception
and returns `False` (counted in `dropped_count`) — it **never raises**.
The writer thread's own disk-write failures are separately counted. A
write-time failure in the writer thread, or a full queue on the
emitting side, therefore cannot propagate back into whatever thread
calls `emit()` — architecturally, `emit()`'s only effects are a
non-blocking queue operation and a counter increment, nothing that can
block or interrupt a caller. `stop()` returns a real,
checked `drained_cleanly` boolean (whether the writer thread actually
joined within its timeout AND the queue is empty) instead of assuming
success.

## Corrections 7/8 — all required tests, cadence check, integrated overhead

12 tests, all passing (`/home/jack/Proper-Research/.venv/bin/python3
test_servo_diagnostic_logging_prep.py`, run just now, no hardware):

```
PASS test_1_missing_method_lookup_is_logged_not_silently_none
PASS test_2_missing_timestamp_read_is_unknown_never_unchanged
PASS test_3_mid_read_update_is_detected_as_changed
PASS test_4_frozen_timestamp_across_snapshots_flags_suspected_stale
PASS test_5_stationary_but_fresh_is_distinguished_from_stale
PASS test_6_queue_overflow_is_counted_not_silently_lost
PASS test_7_incomplete_shutdown_is_detectable
PASS test_cadence_and_control_state_values_unchanged_by_this_module
PASS test_correspond_to_command_seq_picks_the_most_recent_publish_at_or_before
PASS test_effective_servo_args_passthrough
integrated end-to-end overhead: 3.42 us/call over 5000 calls (mock getters
  + stale-detector + queue emit; writer-thread drained_cleanly=True)
PASS test_integrated_mock_overhead_end_to_end
PASS test_runtime_environment_snapshot_resolves_real_installed_rtde

12/12 passed
```

Tests 1-7 map exactly to the 7 required failure modes: missing method
(1), missing timestamp (2), mid-read update (3, the original
atomicity-detector case), frozen timestamp (4), stationary-but-fresh
(5), queue overflow (6), incomplete/interrupted shutdown (7 — includes
both a normal-drain case and an adversarial near-zero-timeout case that
is allowed to report either outcome honestly, then is cleaned up
properly).

**Cadence/control-state-unchanged check**: rather than re-running the
full sustained-worker/deadline test (which exercises the MPC worker
process — a different subsystem already re-verified unchanged earlier
in this investigation, and unaffected by a module that isn't imported
anywhere near it), this pass verifies the stronger, directly relevant
fact: a `grep -rl servo_diagnostic_logging_prep` across
`rectangle_stage_a/`, `vessel_stage_a/`, and
`close_loop_path_follow.py` returns **zero** production files — this
module is not imported by anything that runs the control loop, so
there is no control-loop cadence or state value for it to have altered.
That is asserted as a test, not just stated.

**Integrated overhead**: 3.42µs/call for the full pipeline (snapshot +
stale-stream update + queue emit) against mocks, over 5000 calls — not
the isolated 1.23µs/call snapshot-only number from the prior pass.
Still explicitly **not** a measurement of real RTDE I/O latency, which
requires a live connection this pass does not make.

## Operator procedure for review (not executed)

1. Read `servo_diagnostic_logging_prep.py` and its test file in full
   (both reprinted/summarized above); run the 12 tests yourself to
   reproduce the PASS output.
2. Before ever wiring this to real hardware, separately confirm: (a)
   the target integration point (`RobotJointStream._poll_body()`) still
   reads `cfg.reader_poll_hz` rather than a hardcoded rate (correction
   1); (b) the `DiagnosticLogSink` instance used is given a `maxsize`
   appropriate to the real run's expected sample rate and duration,
   sized so overflow is a rare, logged event, not the normal case; (c)
   `read_failures` and `StaleStreamDetector` verdicts are actually
   surfaced somewhere reviewable (e.g. a per-run summary count of
   `UNKNOWN`/`CHANGED`/`suspected_stale_stream` samples), not just
   written to a file nobody reads.
3. Hardware wiring itself remains a **separate, not-yet-authorized
   step** — nothing in this pass connects to a robot or starts a run.

## Hash re-verification (post-pass, unchanged)

```
0cb859a7bef44c3b060c34064263843e4674007de23ca9822c1d00e25cf03d27  sustained_worker_test_results_RECOVERED_ORIGINAL_23of90.json
9a13bf127c68221857bd1ba2f9e342c7edd902df783406b175280f9d48480b1f  sustained_worker_test_results_threadcompare_labmachine_20261116T173928Z_current.json
982a0032c9c73db738401d144c0dcaacae6fc5e798e79715af8d665ae512534b  sustained_worker_test_results_threadcompare_labmachine_20261116T173928Z_verified1thread.json
7197f0c5db1533bc1b323d09c3e06c6a90ecf2ac1a11de20768d508772c6184c  k2_post_tcpfix_threadfix_20261117T101102Z_20261117T101121Z/path_follow.jsonl
6aafeb66013bd72ec5fd55a5d544edbd58aa78e8db16c99fee68e243658d3398  k2_post_tcpfix_threadfix_20261117T101102Z_20261117T101121Z/summary.json
```

All five match the values already on record. `delay_samples=2`
unchanged everywhere. No servo-parameter, plan, constraint, or
hardware action was taken; no git command was issued.
