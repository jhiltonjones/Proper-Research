# RobotJointStream diagnostic integration — real source changes (2026-11-17)

**No robot connection, no motion, no run execution happened to produce
this document.** No servo tuning, no new excitation, no plan/
`delay_samples` change, no git by the assistant. This is real,
integrated, tested source code in the canonical checkout, as
authorized.

## Files changed

1. `proper_research/hardware/online/state_stream.py` — **modified**
   (`RobotJointStream` class). Neither this file nor
   `servo_diagnostic_logging_prep.py` was ever committed to version
   control, so there is no git diff; sections below show old vs. new
   for every changed region.
2. `proper_research/hardware/online/servo_diagnostic_logging_prep.py` —
   **modified** (split `DiagnosticLogSink`'s single `dropped_count`
   into separate `dropped_count` / `write_failure_count`, per item 5's
   requirement for 3 separately-reportable checks).
3. `proper_research/hardware/online/test_robot_joint_stream_diagnostic_integration.py`
   — **new file** (full content below).

## state_stream.py — old vs. new, by section

**Imports** (added, nothing removed):
```diff
 from proper_research.hardware.online.messages import (
     HeartbeatLoop,
     LatestSlot,
     now_monotonic,
 )
+from proper_research.hardware.online.servo_diagnostic_logging_prep import (
+    DiagnosticLogSink,
+    _safe_optional_call,
+)
 from proper_research.rig_calibration import BEAM_BASE_XYZ_M
```

**`RobotJointStream.__init__`** — old signature/body:
```python
def __init__(
    self, ip: str, *, poll_hz: float = 125.0, receive_only: bool = True
) -> None:
    self._ip = str(ip)
    self._receive_only = bool(receive_only)
    self._robot = None
    self._rtde_r = None
    if self._receive_only:
        import rtde_receive
        self._rtde_r = rtde_receive.RTDEReceiveInterface(ip)
        if not self._rtde_r.isConnected():
            raise RuntimeError(f"rtde_receive could not connect to {ip}")
    else:
        from robot_class import URRtde
        self._robot = URRtde(ip)
    self._poll_hz = float(poll_hz)
    self.joints_slot: LatestSlot[np.ndarray] = LatestSlot()
    self.pose_slot: LatestSlot[np.ndarray] = LatestSlot()
    self._loop: Optional[HeartbeatLoop] = None
    self.reads = 0
    self.read_failures = 0
    self.last_error = ""
    self._last_raw_joints: Optional[np.ndarray] = None
    self._last_raw_pose: Optional[np.ndarray] = None
    self.stale_repeats = 0
```

New: every original line above is **unchanged, in the same order**.
Added: a `_test_rtde_r` injection parameter (test-only — see test file)
inserted *before* the real-connection branch so it takes priority
without altering that branch's own code, plus new diagnostic
parameters (`enable_diagnostic_log`, `diagnostic_log_path`,
`diagnostic_stall_threshold_s`) and a new block of `self.diag_*`
attributes appended strictly after `self.stale_repeats = 0`. Full new
`__init__` is in the repo file; the diagnostic-only tail added is:
```python
        # --- additive, observation-only diagnostic logging ---
        self._diagnostic_enabled = bool(enable_diagnostic_log) and self._receive_only
        self._diag_sink: Optional[DiagnosticLogSink] = None
        if self._diagnostic_enabled and diagnostic_log_path:
            self._diag_sink = DiagnosticLogSink(diagnostic_log_path)
            self._diag_sink.start()
        self._diag_last_ts_value: Optional[float] = None
        self._diag_last_ts_change_host_time: Optional[float] = None
        self._diag_stall_threshold_s = (
            float(diagnostic_stall_threshold_s)
            if diagnostic_stall_threshold_s is not None
            else max(0.5, 10.0 / self._poll_hz)
        )
        self.diag_ticks = 0
        self.diag_internal_errors = 0
        self.diag_duplicate_packets = 0
        self.diag_sustained_stalls = 0
        self.diag_shutdown_drained_cleanly: Optional[bool] = None
```

**`_poll_body()`** — old tail (unchanged, byte-for-byte, still the
last control-relevant statement):
```python
        if pose is not None:
            pose_arr = np.asarray(pose, dtype=float).reshape(6)
            if self._last_raw_pose is None or not np.array_equal(pose_arr, self._last_raw_pose):
                self._last_raw_pose = pose_arr
                self.pose_slot.set(pose_arr)
        self.reads += 1
```
New (appended immediately after, nothing above this point touched):
```python
        if self._diagnostic_enabled:
            try:
                self._diagnostic_tick()
            except Exception:
                self.diag_internal_errors += 1
```

**New method** `_diagnostic_tick()` (full content — reads
`getTimestamp`/`getTargetQ`/`getSpeedScaling`/`getSpeedScalingCombined`
via `_safe_optional_call` on the SAME already-open `self._rtde_r`;
computes 3-state consistency; computes duplicate-vs-stall using real
elapsed host time via `now_monotonic()`; emits to `self._diag_sink` if
configured). See repo file for the exact body (reprinted in the prior
message's tool output and unchanged since).

**`stop()`** — old body (unchanged, byte-for-byte):
```python
    def stop(self) -> None:
        if self._loop is not None:
            self._loop.stop()
            self._loop = None
        if self._robot is not None:
            try:
                self._robot.shutdown()
            except Exception:
                pass
        if self._rtde_r is not None:
            try:
                self._rtde_r.disconnect()
            except Exception:
                pass
```
New (appended after):
```python
        if self._diag_sink is not None:
            self.diag_shutdown_drained_cleanly = self._diag_sink.stop(timeout=2.0)
```

## servo_diagnostic_logging_prep.py — the one change

```diff
-        self.dropped_count = 0
+        self.dropped_count = 0          # queue was full (or emit() itself failed)
+        self.write_failure_count = 0    # writer thread's disk write raised
...
-                except Exception:
-                    self.dropped_count += 1
+                except Exception:
+                    self.write_failure_count += 1
```
(The second change is inside `_writer_loop()`'s disk-write `except`
block — queue overflow and disk-write failure are now two independently
countable things, per item 5.)

## New file (full content)

`proper_research/hardware/online/test_robot_joint_stream_diagnostic_integration.py`
— reprinted in full in this pass's own tool output; summarized by test
name below.

## Offline test command

```
cd /home/jack/Proper-Research
PYTHONPATH=. .venv/bin/python3 proper_research/hardware/online/test_robot_joint_stream_diagnostic_integration.py
```

## Real results (just run, canonical checkout)

```
PASS test_5a_no_dropped_records_under_normal_operation
PASS test_5b_no_write_failures_under_normal_operation
PASS test_5c_shutdown_drains_cleanly
control-path result, logging OFF: {'joints': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 'pose': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 'reads': 1, 'read_failures': 0, 'stale_repeats': 0, 'last_error': '', 'poll_hz': 60.0}
control-path result, logging ON:  {'joints': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 'pose': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 'reads': 1, 'read_failures': 0, 'stale_repeats': 0, 'last_error': '', 'poll_hz': 60.0}
PASS test_control_state_and_cadence_identical_logging_on_vs_off
PASS test_diagnostic_labels_never_touch_existing_stale_repeat_logic
PASS test_duplicate_packet_vs_sustained_stall_by_elapsed_host_time
PASS test_hardware_construction_is_blocked_without_test_injection
PASS test_missing_method_exercised_through_real_integrated_path

8/8 passed
```

Plus the standalone module's own 12 tests, re-run and still passing
after the counter split (`servo_diagnostic_logging_prep.py`'s own
`test_servo_diagnostic_logging_prep.py`, 12/12 passed).

### What each test proves

- **Hardware construction blocked**:
  `test_hardware_construction_is_blocked_without_test_injection` patches
  `rtde_receive.RTDEReceiveInterface` to raise `AssertionError` if ever
  called, then constructs a `RobotJointStream` *without* the
  `_test_rtde_r` injection — proving that path would have hit a real
  connection attempt, and that every OTHER test in this file (which all
  pass `_test_rtde_r`) genuinely never reaches it.
- **Control-path identical, logging on vs. off**:
  `test_control_state_and_cadence_identical_logging_on_vs_off` runs the
  real `_poll_body()` twice (diagnostics off, diagnostics on) against
  identical mock input and asserts the full control-relevant result
  dict (`joints`, `pose`, `reads`, `read_failures`, `stale_repeats`,
  `last_error`, `poll_hz`) is **exactly equal** — shown above as real
  printed output, not just an assertion.
- **Duplicate vs. sustained stall, real elapsed host time**:
  `test_duplicate_packet_vs_sustained_stall_by_elapsed_host_time` shows
  a same-timestamp repeat a few ms later is `duplicate_packet`
  (benign), while the same frozen timestamp after simulated >1s of real
  elapsed host time is `sustained_timestamp_stall`.
- **Diagnostic labels never touch existing safety logic**:
  `test_diagnostic_labels_never_touch_existing_stale_repeat_logic`
  confirms the pre-existing `stale_repeats`/`last_error` (the real
  control-relevant staleness tracking, unchanged in this pass) still
  increments exactly as before, independently of the new `diag_ticks`
  counter.
- **Three separate completeness checks** (item 5): (a) `diag_sink.
  dropped_count == 0`, (b) `diag_sink.write_failure_count == 0`, (c)
  `diag_shutdown_drained_cleanly is True` — three distinct assertions in
  three distinct tests, not one combined pass/fail.
- **Logger failure through the real path**:
  `test_missing_method_exercised_through_real_integrated_path` uses a
  mock that never defines `getSpeedScaling` at all, run through the
  REAL `_diagnostic_tick()` inside `_poll_body()`, confirming
  `diag_internal_errors` stays 0 (guarded, not a crash).

## Operator handoff

**New output file**: when wired with `enable_diagnostic_log=True` and a
real `diagnostic_log_path`, a new JSONL file is written containing one
row per poll tick with:
`host_time_before_s`, `host_time_after_s`, `robot_timestamp_before_raw`,
`robot_timestamp_after_raw`, `target_q_raw_semantics_unverified`,
`speed_scaling`, `speed_scaling_combined`, `within_snapshot_consistency`
(`"changed"`/`"unchanged"`/`"unknown"`), `duplicate_or_stall`
(`"duplicate_packet"`/`"sustained_timestamp_stall"`/`None`), and
`read_failures` (list, empty when nothing failed).

**How to verify it's working correctly, once actually wired to real
hardware (a separate, not-yet-authorized step)**:
- Good signal: `within_snapshot_consistency` is mostly `"unchanged"`
  with occasional `"changed"`; `duplicate_or_stall` is mostly `None`
  or occasional `"duplicate_packet"`; `read_failures` is empty on every
  row; after the run, `RobotJointStream.diag_internal_errors == 0`,
  the sink's `dropped_count == 0` and `write_failure_count == 0`, and
  `diag_shutdown_drained_cleanly is True`.
- Bad signal: any `"sustained_timestamp_stall"` row (the robot-side
  RTDE stream froze for longer than `diagnostic_stall_threshold_s`
  of real elapsed time) — cross-reference against
  `stale_repeats`/`last_error` (the pre-existing control-relevant
  staleness counter) for the same window; a nonzero `dropped_count` or
  `write_failure_count`; or `diag_shutdown_drained_cleanly is False`
  (incomplete shutdown — some logged rows may be missing from the
  file).
- This logging **observes only** — nothing above changes what the
  control loop measures, decides, or commands. Wiring
  `enable_diagnostic_log=True` into any real run script, and
  connecting to real hardware, remain separate, not-yet-authorized
  steps.

## Hash re-verification (post-pass, unchanged)

```
0cb859a7bef44c3b060c34064263843e4674007de23ca9822c1d00e25cf03d27  sustained_worker_test_results_RECOVERED_ORIGINAL_23of90.json
9a13bf127c68221857bd1ba2f9e342c7edd902df783406b175280f9d48480b1f  sustained_worker_test_results_threadcompare_labmachine_20261116T173928Z_current.json
982a0032c9c73db738401d144c0dcaacae6fc5e798e79715af8d665ae512534b  sustained_worker_test_results_threadcompare_labmachine_20261116T173928Z_verified1thread.json
7197f0c5db1533bc1b323d09c3e06c6a90ecf2ac1a11de20768d508772c6184c  k2_post_tcpfix_threadfix_20261117T101102Z_20261117T101121Z/path_follow.jsonl
6aafeb66013bd72ec5fd55a5d544edbd58aa78e8db16c99fee68e243658d3398  k2_post_tcpfix_threadfix_20261117T101102Z_20261117T101121Z/summary.json
```

All five match the values already on record. `delay_samples=2`
unchanged everywhere. No servo-parameter, plan, or hardware action was
taken; no git command was issued.
