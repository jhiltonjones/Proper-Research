# Servo diagnostic logging — integration proposal (review only)

**No hardware connection and no run of any kind happened to produce
this document.** No controller/servo-parameter/plan/`delay_samples`
change, no new excitation signal, no git by the assistant. This
consolidates the previously prepared module/tests/procedure (reprinted
below) with explicit, honest answers to the 5 pre-wiring questions. It
is a proposal for the user's review, not an authorization to run
anything.

## The module (reprinted)

`proper_research/hardware/online/servo_diagnostic_logging_prep.py` —
not imported by any production run script.

```python
def runtime_environment_snapshot() -> dict:
    """No hardware access. Pure interpreter/module introspection."""
    out: dict = {
        "python_executable": sys.executable,
        "sys_path_head": sys.path[:3],
    }
    for modname in ("rtde_control", "rtde_receive"):
        try:
            mod = __import__(modname)
            out[f"{modname}_file"] = getattr(mod, "__file__", None)
        except Exception as exc:
            out[f"{modname}_file"] = None
            out[f"{modname}_import_error"] = repr(exc)
    try:
        import importlib.metadata as _im
        out["ur_rtde_version"] = _im.version("ur_rtde")
    except Exception as exc:
        out["ur_rtde_version"] = None
        out["ur_rtde_version_error"] = repr(exc)
    return out


def effective_servo_args(time_s, lookahead_time, gain) -> dict:
    return {
        "servo_time_s": float(time_s),
        "servo_lookahead_time_s": float(lookahead_time),
        "servo_gain": int(gain),
    }


@dataclass
class RobotSideSnapshot:
    timestamp_before: Optional[float]
    timestamp_after: Optional[float]
    actual_q: Optional[list]
    target_q_raw_semantics_unverified: Optional[list]   # name carries the caveat
    speed_scaling: Optional[float]
    speed_scaling_combined: Optional[float]
    atomic_suspect: bool = field(init=False)

    def __post_init__(self):
        self.atomic_suspect = (
            self.timestamp_before is not None
            and self.timestamp_after is not None
            and self.timestamp_before != self.timestamp_after
        )


def robot_side_snapshot(rtde_r) -> RobotSideSnapshot:
    def _safe(fn):
        try:
            return fn()
        except Exception:
            return None
    ts_before = _safe(rtde_r.getTimestamp)
    actual_q = _safe(rtde_r.getActualQ)
    target_q = _safe(rtde_r.getTargetQ)
    speed_scaling = _safe(rtde_r.getSpeedScaling)
    speed_scaling_combined = _safe(rtde_r.getSpeedScalingCombined)
    ts_after = _safe(rtde_r.getTimestamp)
    return RobotSideSnapshot(ts_before, ts_after, actual_q, target_q,
                              speed_scaling, speed_scaling_combined)


def correspond_to_command_seq(snapshot_t_s, publish_events) -> Optional[int]:
    candidates = [e for e in publish_events
                  if e.get("publish_time_s", 1e18) <= snapshot_t_s]
    if not candidates:
        return None
    return max(candidates, key=lambda e: e["publish_time_s"]).get("seq")
```

## The six tests (reprinted, all passing against mocks, no hardware)

```python
test_runtime_environment_snapshot_resolves_real_installed_rtde()
    # asserts python_executable, rtde_control_file, ur_rtde_version all set

test_effective_servo_args_passthrough()
    # effective_servo_args(0.04, 0.2, 200) == exact dict, no transformation

test_robot_side_snapshot_flags_atomic_case_as_not_suspect()
    # mock returns the SAME getTimestamp() both times -> atomic_suspect is False

test_robot_side_snapshot_detects_non_atomic_update_mid_snapshot()
    # mock returns a DIFFERENT getTimestamp() the 2nd time (simulates a
    # background update landing mid-snapshot) -> atomic_suspect is True
    # *** this is the test that proves the detector actually catches the
    # failure mode, rather than assuming it away ***

test_robot_side_snapshot_tolerates_a_failing_getter()
    # one getter raises -> that field becomes None, others unaffected,
    # no crash

test_correspond_to_command_seq_picks_the_most_recent_publish_at_or_before()
    # exact-match, before-any-publish, and tie-at-boundary cases
```

All six: `PASS` (re-run for this document;
`/home/jack/Proper-Research/.venv/bin/python3
test_servo_diagnostic_logging_prep.py`, no hardware required).

## Answers to the 5 pre-wiring questions

### 1. Connection: zero additional, by reusing an existing one

Traced the real connection topology in `close_loop_path_follow.py`'s
real run path: it already opens **two** separate RTDE sessions
simultaneously — `URRTDERobot` (`ur_rtde_robot.py:82-108`, a combined
`rtde_control.RTDEControlInterface` + `rtde_receive.RTDEReceiveInterface`
pair, used for `servo_j()`), and a second, independent, receive-only
`rtde_receive.RTDEReceiveInterface` owned by `RobotJointStream`
(`state_stream.py:151-175`, `receive_only=True`, used as the `reader`
object whose `latest_joints_with_timestamp()` already feeds
`q_meas_rad` into every logged row — confirmed at
`close_loop_path_follow.py:1292`). **The proposal is to extend that
SECOND, already-open, already-polled-at-125Hz connection** (`reader`'s
`self._rtde_r`) with the 3 additional getter calls
(`getTargetQ`/`getSpeedScaling`/`getSpeedScalingCombined`/`getTimestamp`)
inside its existing `_poll_body()` (`state_stream.py:230+`) — **zero new
RTDE connections**. This is not assumed costless in general (UR
controllers do cap concurrent RTDE clients, and this project already
runs 2 of whatever that cap is), but reusing an existing connection
specifically avoids adding a 3rd.

### 2. Field-by-field atomicity — real answer, not a blanket claim

Checked the installed library's header (`rtde_receive_interface.h:500-503`):
`std::atomic<bool> stop_receive_thread`, `std::shared_ptr<boost::thread>
th_` — confirms a single background thread continuously receives RTDE
packages and updates one shared `RobotState` struct; the Python getters
(`getActualQ`, `getTargetQ`, `getSpeedScaling`, `getTimestamp`, etc.)
are field reads off that cached struct, **not** separate network round
trips. `actual_q`, `target_q`, `speed_scaling`, and `timestamp` are all
part of the SAME RTDE output-variable subscription, so **within one
incoming package** they are updated together by that one background
thread write. **But**: the exposed Python API gives no "freeze the
struct, read N fields, release" transaction — each getter call is an
independent read of whatever the struct currently holds. So: in the
common case (Python-side getter calls take ~1-2us each per this
pass's own measurement, vs. an RTDE cycle on the order of several ms),
the probability of the background thread updating *between* two of
these specific getter calls is low, but the API gives **no guarantee**
— there is a real, nonzero window in which `actual_q` could reflect
package N while `target_q` read microseconds later reflects package
N+1. This is exactly, and only, what `atomic_suspect` is designed to
flag (via the `getTimestamp()` bracket) — not a blanket "it's fine" or
"it's broken," but a per-sample, evidence-based flag.

### 3. What the detector guarantees, what it cannot catch, how flagged

**Guarantees**: if the background thread updates the shared state
between the `getTimestamp()` call made immediately before the other 4
getters and the one made immediately after, `timestamp_before !=
timestamp_after` and `atomic_suspect` is set `True` on that sample.
This *does* catch "the underlying state changed somewhere during this
snapshot's window."

**Cannot catch**: (a) it cannot identify *which* of the 4 bracketed
fields (if any) is the stale one versus the fresh one — only that a
change happened somewhere in the window, not where; (b) if the update
happens to leave `getTimestamp()` returning an identical value by
coincidence (not expected given RTDE timestamps are monotonic
doubles, but not something this pass has proven impossible); and (c)
— the most concrete, code-grounded gap — this project's own
`state_stream.py` already documents a real failure mode
(`state_stream.py:186-196,208-215`, the 2026-09-17 "stale_repeat" bug)
where a stalled RTDE stream keeps returning the exact same cached
packet indefinitely with no exception. In that failure mode,
`getTimestamp()` would also return the identical value on both
bracketing calls — `atomic_suspect` would read `False`, which is
indistinguishable, by this detector alone, from genuine real-time
freshness. **The detector answers "did anything change during this
specific snapshot," not "is this data currently fresh"** — that
second, different question is already covered by this project's
existing stale-repeat check on the raw joint/pose readings
(`state_stream.py`'s `_last_raw_joints`/`stale_repeats` mechanism) and
should be cross-referenced, not re-solved here.

**User-visible flag**: `RobotSideSnapshot.atomic_suspect` (bool), plus
the raw `timestamp_before`/`timestamp_after` pair logged alongside it
so a reviewer can see the actual values, not just the boolean.

### 4. Rate, overhead, shutdown

**Rate**: proposed tied to the *existing* `RobotJointStream` poll rate
(125Hz / ~8ms, `state_stream.py`'s own `poll_hz` default) — not a new,
independent cadence, and not tied to the 20ms streamer cycle (that is
a different thread/connection entirely, per item 1's trace).

**Overhead, measured not guessed**: `timeit`-benchmarked
`robot_side_snapshot()` against the mock in this pass:
**1.23 microseconds per call** over 20,000 calls. This is the
Python-side dataclass/function-call overhead only — it measures
dispatch to 5 mock getters that return instantly. **It does not and
cannot measure real RTDE I/O latency**, since no hardware connection
exists in this pass; the real per-call cost would be dominated by
whatever `getActualQ()`/`getTargetQ()`/etc. actually cost against the
real background-thread-cached struct (expected to be similarly small,
since those are cached-field reads, not network calls per the item-2
tracing — but this expectation is not empirically confirmed here).

**Shutdown**: proposed to route through the *same* queue-based
writer-thread pattern `BackgroundArmStreamer` already uses
(`close_loop_path_follow.py:151-179`, `enable_event_log`/`_writer_loop`)
— the real-time thread only does a non-blocking `queue.put_nowait()`,
a separate writer thread drains to disk, and `stop()`
(`close_loop_path_follow.py:295-303`) already `join()`s that writer
thread with a timeout before returning. Reusing this exact,
already-proven-clean-draining mechanism for a new event type is the
proposal — not a new, unverified shutdown path.

### 5. Sequence correspondence without treating `target_q` as acknowledgement

`correspond_to_command_seq()` uses only **host-side wall-clock timing**:
`BackgroundArmStreamer.update_target()` already timestamps every
publish (`_target_publish_time`, `close_loop_path_follow.py:194-207`,
tagged with its own `seq`). If a robot-side snapshot also records a
host-side monotonic timestamp bracketing its RTDE getter calls, then
"this snapshot was taken at host time T" can be matched to "the most
recent published `seq` at or before T" — establishing **only** a
temporal ordering ("this snapshot was taken after command N was sent
and before N+1 was sent"), never a claim that the robot's `target_q`
field *equals* command N's value or that the robot has acknowledged
receiving it. `target_q`'s own value is logged purely for its own
sake, separately, still under its
`target_q_raw_semantics_unverified` name — the correspondence
mechanism does not read or depend on that value at all.

## Interpreter/version — from the actual process, stated honestly

Everything in this pass's own verification (the `.venv` resolution,
the `1.6.2` version, the 1.23us benchmark) was run against the
**actual installed `.venv` interpreter and the mock test harness** —
real process, real library, but no real robot connection. **This
cannot and does not claim anything about which `ur_rtde` version any
PAST hardware run (including the K2 run analyzed throughout this
investigation) actually used** — no past run's own logs record an
`ur_rtde` version (confirmed: `controller_metadata.json`/`summary.json`
have no such field). That is unknown for every past run and can only
be settled going forward once `runtime_environment_snapshot()` is
actually wired into a run's startup logging — which has not happened.

## Explicit control-independence confirmation

This logging, as designed and tested here, **only observes**. It does
not substitute for any measured state anywhere in the control path, it
introduces no new excitation signal, and it does not touch
`servo_lookahead_s`, `servo_gain`, any plan file, `delay_samples`, or
any constraint. Nothing the controller decides is read from, or
influenced by, anything this module produces.

## Bounded operator procedure (for review — not executed)

1. Add 4 getter calls (`getTargetQ`, `getSpeedScaling`,
   `getSpeedScalingCombined`, `getTimestamp` x2) inside
   `RobotJointStream._poll_body()`'s existing `receive_only` branch —
   same connection, no new session.
2. Route the resulting `RobotSideSnapshot` (plus a host-side monotonic
   timestamp) through a new event type in the existing
   `queue.Queue`/writer-thread pattern already proven in
   `BackgroundArmStreamer`.
3. Call `runtime_environment_snapshot()` once at startup, written into
   the existing `controller_metadata.json`-style record.
4. Offline, post-run: call `correspond_to_command_seq()` against the
   already-logged `publish` events to annotate each snapshot with its
   nearest preceding command `seq`, and cross-reference
   `atomic_suspect`-flagged and `state_stream.py`'s own stale-repeat
   counter together before trusting any sample.
5. **Requires separate, explicit approval before step 1 touches any
   file that runs against real hardware.** Nothing above is applied in
   this pass.

## Hash re-verification (post-pass, unchanged)

```
0cb859a7bef44c3b060c34064263843e4674007de23ca9822c1d00e25cf03d27  sustained_worker_test_results_RECOVERED_ORIGINAL_23of90.json
9a13bf127c68221857bd1ba2f9e342c7edd902df783406b175280f9d48480b1f  sustained_worker_test_results_threadcompare_labmachine_20261116T173928Z_current.json
982a0032c9c73db738401d144c0dcaacae6fc5e798e79715af8d665ae512534b  sustained_worker_test_results_threadcompare_labmachine_20261116T173928Z_verified1thread.json
7197f0c5db1533bc1b323d09c3e06c6a90ecf2ac1a11de20768d508772c6184c  k2_post_tcpfix_threadfix_20261117T101102Z_20261117T101121Z/path_follow.jsonl
6aafeb66013bd72ec5fd55a5d544edbd58aa78e8db16c99fee68e243658d3398  k2_post_tcpfix_threadfix_20261117T101102Z_20261117T101121Z/summary.json
```

All five match the values already on record. No hardware connection,
no run, no controller/servo/plan/`delay_samples` change, no git. Day-1
stays blocked.
