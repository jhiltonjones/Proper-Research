# Corrections to the servoJ command-path trace (2026-11-17)

Strictly read-only investigation + tested (mocks-only, no hardware
motion) logging preparation. No controller/servo-parameter/plan/
`delay_samples` change, no hardware motion, no git by the assistant.

## 1. "Replace, not queue" — withdrawn as previously stated

The prior claim was built from `strings` output on the compiled
`librtde.so`, which is a flat list of embedded text, not a disassembly
or trace of actual control flow — it cannot by itself establish real
timing, locking, or whether writes between reads are coalesced. That
claim is withdrawn as previously stated.

**Version resolved concretely, not left as "two installed, ambiguous"**:
the project's own documented procedure
(`K2_POST_TCPFIX_THREADFIX_OPERATOR_PROCEDURE.md:13,72`: `source
.venv/bin/activate`) determines this. Checked `.venv/bin/python3`'s
actual `sys.path` directly: it contains
`/home/jack/Proper-Research/.venv/lib/python3.10/site-packages` and
**does not contain** `~/.local/lib/python3.10/site-packages` at all
(`.venv/pyvenv.cfg` has `include-system-site-packages = false`, and
user-site isn't added either once the venv's own `site-packages` is
authoritative). Confirmed by direct import: `.venv/bin/python3 -c
"import rtde_control; print(rtde_control.__file__)"` resolves to
`.venv/lib/python3.10/site-packages/rtde_control...so`, and
`importlib.metadata.version('ur_rtde')` under that same interpreter
reports **1.6.2**. So: when the project's own documented activation
step is followed, there is no ambiguity — **1.6.2** is what resolves.
(The `~/.local` 1.6.0 install is simply unreachable from that
environment; it would only matter if someone ran a bare
`/usr/bin/python3` without activating the venv, which is not what the
project's own procedure instructs.)

**What the embedded strings DO and DON'T support, now scoped
honestly**: the extracted text contains `global servo_target = [0, 0,
0, 0, 0, 0]` — a single global variable, not an array/queue — which is
real, if limited, evidence that the robot-side script's *storage* for
the pending target is a single overwritable slot, not a FIFO. That is
as far as this evidence goes. It does **not** establish: the real
read/write locking discipline between the register-write and the
`servo_thread`'s own read of that global, the real polling frequency of
that thread relative to incoming register writes, or whether a write
that lands between two robot-side polls is ever dropped or merged
rather than simply overwritten-then-read. Those require either
disassembly (not done) or a live, instrumented test against real
hardware (not done, and not proposed here). **Open gap, not resolved.**

## 2. Three distinct rates — kept separate, not inferred from each other

- **(a) New MPC targets**: 100ms, the control loop's own cadence
  (`PathFollowConfig`'s control period; confirmed elsewhere in this
  investigation, unchanged here).
- **(b) Host-side register writes**: 20ms
  (`_STREAM_INTERVAL_S = 0.02`, `close_loop_path_follow.py:85`) — the
  streamer's `_run()` loop calls `servo_j()` every 20ms regardless of
  whether the underlying target changed; `arm_streamer_events.jsonl`'s
  own `is_new_target` field (already used in the pickup-latency
  correction) directly confirms most of these are repeats of the same
  MPC-tick's target (1000 of 1250 logged pickups were repeats, 250 were
  genuinely new — a 4:1 ratio, consistent with 100ms/20ms=5 minus
  boundary effects).
- **(c) Robot-side consumption rate**: **not determined.** Nothing in
  the `ur_rtde` Python API, its documentation, or the extracted strings
  gives the real-time period of the robot controller's own internal
  `servo_thread`/`servoj()` execution cycle. This is stated as an open
  gap, not inferred from (a) or (b) — the robot controller's real-time
  loop runs on its own clock (UR controllers commonly run an internal
  500Hz-class real-time loop, but nothing inspected in this checkout
  confirms that figure for this specific controller/firmware, so it is
  not asserted here).

## 3. Logging claim corrected — `summary.json` already has this

Checked `summary.json` directly (not reused from memory): it contains,
under `.config`:
```
"servo_lookahead_s": 0.2,
"servo_gain": 200,
"servo_stream_hz": 50.0,
```
**The prior trace's "not logged anywhere" claim was wrong — withdrawn.**
These values are recorded once per run (as the serialized
`PathFollowConfig`), confirming the actual lookahead/gain used (0.2s,
200) match the source-code defaults traced earlier, with real evidence
from the run's own output, not just from reading the source.

**Gain range, corrected**: the library's own doc
(`/usr/include/ur_rtde/rtde_control_interface_doc.h` / confirmed again
via `rtde_control.RTDEControlInterface.servoJ.__doc__`) states gain's
valid range is **[100, 2000]**. 200 is **100 above the minimum of a
1900-wide range — ~5.3% into it**, i.e. **near the lower end**, not
"mid-range" as the prior trace stated (mid-range would be ~1050). That
prior characterization is corrected here.

## 4. Additive diagnostic logging — prepared and tested, NOT wired into production

Written as a new, standalone module
(`proper_research/hardware/online/servo_diagnostic_logging_prep.py`),
not imported by any production run script, plus a test file
(`test_servo_diagnostic_logging_prep.py`) exercising it against mocks
only. All 6 tests pass (no hardware involved):

- `runtime_environment_snapshot()`: logs `sys.executable`, the resolved
  `rtde_control`/`rtde_receive` module file paths, and
  `importlib.metadata.version('ur_rtde')` — settles item 1's version
  question automatically on every future run instead of requiring
  manual tracing. Tested against the real installed library (test
  `test_runtime_environment_snapshot_resolves_real_installed_rtde`,
  passes, confirms it would log `.venv`'s 1.6.2 under the documented
  activation procedure).
- `effective_servo_args()`: trivial per-call passthrough of
  t/lookahead/gain — redundant with the per-run `summary.json` value
  given item 3's finding, but available per-call if a future need to
  confirm no mid-run override arises.
- `robot_side_snapshot()`: the one requiring real care per the
  atomicity requirement. **Checked `rtde_receive`'s actual header**
  (`/usr/include/ur_rtde/rtde_receive_interface.h`): `RTDEReceiveInterface`
  is backed by a `RobotState` object populated by a background thread
  (confirmed by the `#include <ur_rtde/robot_state.h>` and `<atomic>`
  includes), and the Python API exposes **no** explicit "receive once,
  then read many fields transactionally" call — only individual getters
  plus a `reconnect()`. **Sequential getter calls are therefore not
  assumed atomic.** The function calls `getTimestamp()` immediately
  before and after the other four getters and exposes both as
  `timestamp_before`/`timestamp_after`, with `atomic_suspect = True`
  whenever they differ — i.e. non-atomicity is *detected*, not assumed
  away. Tested against three mocks: a case where all fields genuinely
  share one timestamp (`atomic_suspect` correctly `False`), a case
  built specifically to simulate a background update landing mid-
  snapshot (`atomic_suspect` correctly `True` — this is the test that
  matters: it proves the design catches the failure mode the user
  flagged, rather than silently missing it), and a case with a failing
  getter (degrades that one field to `None` without crashing the
  others). The `getTargetQ()` field is named
  `target_q_raw_semantics_unverified` in the dataclass itself — the
  caveat about its unknown raw-echo-vs-smoothed-setpoint semantics is
  carried into the field name, not left in a comment that can be lost
  on reuse.
- `correspond_to_command_seq()`: ties a robot-side snapshot's wall-clock
  time back to the streamer's existing `publish` event `seq` numbers
  (already logged, `close_loop_path_follow.py:181-207` — no new field
  needed on that side). Tested with three cases (exact match, before
  any publish, and tie-at-boundary).

**Nothing here is wired into `close_loop_path_follow.py`,
`BackgroundArmStreamer`, or any run script.** It is a reviewed,
unit-tested, standalone module only.

## Bounded operator procedure (for review, not executed)

If the user wants this logging live on a future run:
1. Call `runtime_environment_snapshot()` once at run startup (no
   hardware dependency) and write its dict into the existing
   `controller_metadata.json`-style startup record.
2. Call `robot_side_snapshot(rtde_r)` from inside
   `BackgroundArmStreamer._run()` (or a sibling low-rate thread) using
   the project's existing `rtde_receive` handle, at a rate no higher
   than the existing 20ms streamer cadence, writing each snapshot
   (including its `atomic_suspect` flag) to a new event type in the
   existing per-cycle event log (`arm_streamer_events.jsonl`'s
   existing writer-thread/queue pattern, so file I/O stays off the
   real-time thread, matching the project's own established
   convention).
3. Call `correspond_to_command_seq()` offline, during analysis, not
   inside the hot loop.
4. **Explicitly requires separate approval** before any of this runs
   against the real robot — this document proposes and tests the
   logging in isolation only.

## Remaining uncertainty, stated plainly

- Robot-side target-consumption rate (3c above): unknown, not
  inferable from what's installed or documented here.
- `getTargetQ()`'s true semantics (raw echo vs. internally-smoothed
  setpoint): unknown; the prepared logging's field name carries this
  caveat forward rather than resolving it.
- Whether the host-side register write and the robot-side global
  `servo_target` read are themselves properly synchronized (vs. a
  benign race the controller's real-time scheduler happens to make
  irrelevant in practice): not established from available evidence.

## Hash re-verification (post-pass, unchanged)

```
0cb859a7bef44c3b060c34064263843e4674007de23ca9822c1d00e25cf03d27  sustained_worker_test_results_RECOVERED_ORIGINAL_23of90.json
9a13bf127c68221857bd1ba2f9e342c7edd902df783406b175280f9d48480b1f  sustained_worker_test_results_threadcompare_labmachine_20261116T173928Z_current.json
982a0032c9c73db738401d144c0dcaacae6fc5e798e79715af8d665ae512534b  sustained_worker_test_results_threadcompare_labmachine_20261116T173928Z_verified1thread.json
7197f0c5db1533bc1b323d09c3e06c6a90ecf2ac1a11de20768d508772c6184c  k2_post_tcpfix_threadfix_20261117T101102Z_20261117T101121Z/path_follow.jsonl
6aafeb66013bd72ec5fd55a5d544edbd58aa78e8db16c99fee68e243658d3398  k2_post_tcpfix_threadfix_20261117T101102Z_20261117T101121Z/summary.json
```

All five match the values already on record. `delay_samples=2`
unchanged everywhere. No controller, servo-parameter, plan, or
hardware action was taken. No PRBS, no criterion approval, no
production wiring is authorized by this document.
