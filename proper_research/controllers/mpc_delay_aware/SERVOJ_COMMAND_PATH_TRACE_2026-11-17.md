# servoJ command-path trace (2026-11-17)

**SUPERSEDED IN PART.** The user found real problems: the "replace, not
queue" claim was not fully demonstrated (strings-only evidence, no
control-flow trace); the two-installed-versions ambiguity was left
unresolved when it is actually resolvable (the project's own documented
`.venv` activation determines it); `lookahead_time`/`gain` ARE logged in
`summary.json` (the prior "not logged anywhere" claim was wrong); and
`gain=200` is near the LOW end of `[100,2000]`, not mid-range. See
`SERVOJ_COMMAND_PATH_TRACE_CORRECTIONS_2026-11-17.md` for the corrected
findings, the three-rates distinction, and tested (mocks-only)
diagnostic-logging preparation. This file is kept unmodified below as
the historical record — do not cite its "replace, not queue," version-
ambiguity, or logging-gap claims without reading the corrections file.

Strictly read-only investigation. No controller/protocol change, no new
motion, no PRBS implementation, no git by the assistant. All claims
below are cited to real file:line or real library doc/binary evidence;
anything not directly verifiable is stated as an open gap, not assumed.

## Complete path, traced

1. **`BackgroundArmStreamer._run()`** —
   `proper_research/hardware/online/close_loop_path_follow.py:223-293`.
   A dedicated thread, paced by its own `next_cycle += 0.02` scheduler
   (`_STREAM_INTERVAL_S = 0.02`, line 85), re-reads whatever target the
   main control loop last handed off via `update_target()` and calls
   `self._robot.servo_j(target, time_s=..., lookahead_time=...,
   gain=...)` every 20ms (line 259-264).
2. **`URRTDERobot.servo_j()`** wrapper —
   `proper_research/hardware/ur_rtde_robot.py:348-378`. Validates
   `gain` is in `[100, 2000]`, then calls
   `self._control.servoJ(target, speed, acceleration, time_s,
   lookahead_time, gain)` (line 373-377), where `self._control` is an
   `rtde_control.RTDEControlInterface` instance (constructed from
   `import rtde_control` at line 15).
3. **Installed library**: `ur_rtde`. Two installations exist in this
   checkout: `/home/jack/.local/lib/python3.10/site-packages`
   (`ur_rtde-1.6.0.dist-info`, found via `import ur_rtde` metadata
   lookup — `ur_rtde 1.6.0`) and
   `/home/jack/Proper-Research/.venv/lib/python3.10/site-packages`
   (`ur_rtde-1.6.2.dist-info`). The project's own run scripts use a
   `#!/usr/bin/env python3` shebang with no explicit interpreter pin
   visible in the script header, so which one a given invocation
   actually loads depends on which `python3` is first on `PATH` at run
   time — **not independently re-derivable from this trace alone**;
   both installations' compiled `servoJ` docstrings are textually
   identical (checked both: see item 2 below), so this ambiguity does
   not affect the behavioral claims here, but the exact version used
   for any specific past run is not something this trace can pin down
   without knowing that invocation's `PATH`/venv activation state.
   Neither version is pinned in `environment.yml` (`grep -i rtde
   environment.yml` returns nothing) — `ur_rtde` is a pip-only
   dependency, not declared in the project's own manifest, the same
   gap pattern `threadpoolctl` had before its adoption.
4. **Robot-side execution**: `strings` on the installed
   `librtde.so.1.6.2` (the C++ runtime `rtde_control`'s compiled
   extension links against) contains literal embedded URScript text,
   including:
   ```
   thread servo_thread():
   thread servoc_thread():
       while (True):
           target_waypoint = get_target_waypoint()
           ...
           servoj(q, t=dt, lookahead_time=lh_time, gain=g)
   ```
   This confirms the mechanism directly from the shipped binary, not
   from general background knowledge: `ur_rtde` uploads one persistent
   control script to the robot controller (confirmed separately by the
   library's own documented `reuploadScript()` method,
   `/usr/include/ur_rtde/rtde_control_interface_doc.h:529-532`: "This
   will only happen if a script is not already running on the
   controller"). That script runs its own `servo_thread` loop calling
   the UR controller's native `servoj()` motion primitive continuously,
   reading the current commanded target via `get_target_waypoint()`
   each iteration — which is fed by whatever the host's last
   `rtde_c.servoJ()` call wrote into the RTDE register interface.

## The 5 confirmation items

**1. Registers vs. script socket.** Registers, confirmed by the
`get_target_waypoint()`/register-write pattern above and by
`reuploadScript`'s documented behavior (one persistent script, not a
fresh script-socket upload per command). The host-side `servoJ()` call
writes the new target into the RTDE register interface; it does not
open a new script-socket connection per call.

**2. Real `t`/`lookahead_time`/`gain` at the actual call site.** Traced
to the real, unoverridden config defaults:
`PathFollowConfig.servo_lookahead_s: float = 0.20` and `servo_gain: int
= 200` (`close_loop_path_follow.py:617-618`), passed straight through
to `BackgroundArmStreamer(..., lookahead_time=cfg.servo_lookahead_s,
gain=cfg.servo_gain)` (`close_loop_path_follow.py:1942-1943`) with
`time_s=2.0 * _STREAM_INTERVAL_S` = **0.04s** (line 1941, comment at
1933-1938 explains the 2x-stream-interval convention). Confirmed the
K2 run's actual `argv` (`controller_metadata.json`) passes no
`--servo-lookahead`/`--servo-gain` flags and the run script
(`run_mpc_delay_aware_insertion_anchor.py`) defines no such override —
only `--tcp-workspace-robust-margin-mm 15` and similar were set — so
the config defaults are what actually reached the call: **t=0.04s,
lookahead_time=0.20s, gain=200**. These were **not** logged anywhere in
`path_follow.jsonl`/`controller_metadata.json` for this run (confirmed
by field listing) — they can only be recovered by reading the source
that was live at run time, which is what this trace did.

**3. Blocking vs. return-immediately — real discrepancy found.** The
installed library's own compiled docstring (identical in both 1.6.0
and 1.6.2, and matching the public C++ header
`/usr/include/ur_rtde/rtde_control_interface_doc.h:594-616` verbatim)
states: *"time: time where the command is controlling the robot. The
function is blocking for time t \[S\]."* That is the library's own
documented claim — `servoJ` should block the calling (host) thread for
`t` seconds (here, 0.04s). **But the real measured behavior on this
run's own hardware log contradicts that**: `arm_streamer_events.jsonl`
logs `servo_duration_ms` for every one of 1250 real `servo_j()` calls
(`close_loop_path_follow.py:273-281`), and the measured distribution is
mean 0.09ms, p99 0.20ms, max 0.27ms — **three orders of magnitude
below the documented 40ms block**. Given the embedded-URScript
evidence in item 1/4 (the robot-side loop, not the host call, runs the
40ms-scale `servoj()`/`lookahead_time` behavior), the most
code-consistent explanation is that the *documented* "blocking"
describes the robot-controller-side script loop's own cadence, while
the Python-side RTDE register write (what this project's host
actually measures as `servo_duration_ms`) returns as soon as the write
completes — but this project's logs do not and cannot directly prove
*why* the documented claim and the measured host-side timing diverge
this much; stating that explanation with certainty would be guessing
past what's actually traceable from this checkout. What is directly
measured, with real evidence: the host call does **not** block for
anywhere near `time_s`.

**4. 20ms host updates vs. 40ms `time_s` — real mechanism.** Per the
embedded `servo_thread`/`get_target_waypoint()` loop (item 4 above),
the robot-side script re-reads the current register target on its own
internal cycle and immediately feeds it to the native `servoj()`
primitive — it is a **replace, not a queue**: there is one register
holding "the current target," and each `update_target()`→`servo_j()`
handoff overwrites it. Since the host streams a fresh target every
20ms while `time_s=0.04s` (40ms) is the duration parameter passed to
each call, a new target arrives, on average, **before** the previous
call's own `time_s` window would have elapsed — i.e. the robot's
internal trajectory is continuously re-targeted every ~20ms, and
`time_s=0.04` functions as (per the library's own doc) "time where the
command is controlling the robot" for a call that in practice gets
superseded by a newer one roughly halfway through that window, not run
to completion. This is consistent with, and is exactly the documented
purpose of, `lookahead_time` "smoothen\[ing\] the trajectory" between
successive overlapping targets — but this trace cannot quantify the
resulting net effective lag from first principles; that would require
either the UR controller firmware's internal `servoj()` implementation
(not available here — it is proprietary controller-side code, not
part of the `ur_rtde` package) or a live measurement of
`getTargetQ()` vs. `getActualQ()` (see "remaining gap" below, not
currently logged).

**5. Speed scaling / UR controller settings.** `rtde_receive` exposes
`getSpeedScaling()` and `getSpeedScalingCombined()` (confirmed present
via `dir(rtde_receive.RTDEReceiveInterface)` on the installed 1.6.2
build) — **neither is called anywhere in this project**
(`grep -rn "getSpeedScaling\|speed_scaling" proper_research/` returns
nothing). This is a real, confirmed gap, not an assumption: **no run
analyzed anywhere in this investigation has ever logged the robot's
speed-scaling/slider state.** It would be wrong to assume 100%
(unscaled) held throughout every run — that was never checked or
recorded, and this trace found no mechanism in the code that would
have forced it.

## `lookahead_time=0.20` / `gain=200` — assessed, not changed

The installed library's own documentation (both versions, matching the
public header) describes `lookahead_time` as: *"time \[S\], range
\[0.03, 0.2\], smoothens the trajectory with this lookahead time"* —
explicitly a **trajectory-smoothing window**, not a fixed transport
delay. 0.20s is the *maximum* of the documented valid range, i.e. this
project is already running the most aggressive smoothing the parameter
allows. `gain` (200, mid-range of the documented `[100, 2000]`) is
described as "proportional gain for following target position" — a
closed-loop tracking gain on the robot's own internal position
servo, not a time quantity at all.

**Do not equate `lookahead_time=0.2` with "200ms of delay."** What the
documentation and the embedded-script evidence together support is
more nuanced: both parameters jointly shape the ROBOT CONTROLLER's own
internal closed-loop response to a continuously-updated target
(how smoothly/tightly it blends between successive 20ms-spaced
targets), which **could plausibly contribute some fraction of an
effective lag** to the plant's realized response — consistent with
this investigation's repeated finding of a multi-sample (~2-3 tick)
delay — but neither parameter is a literal fixed-latency constant, and
nothing in the available documentation or this project's logs lets
this trace assign a specific millisecond value to that contribution.
That would require either the proprietary controller-side `servoj()`
implementation or a direct `getTargetQ()`-vs-`getActualQ()`
measurement (next section) — neither of which exists in this checkout.

## Remaining measurement gap: minimum additional instrumentation needed

To separate (a) command delivery, (b) the robot's own internal target
response, and (c) measured joint motion, the minimum addition would be
logging `rtde_receive.RTDEReceiveInterface.getTargetQ()` (confirmed to
exist: "Returns: Target joint positions") alongside the already-logged
`getActualQ()`-derived `q_meas_rad`, at the same tick.

**Caveat, not assumed**: the installed library's one-line doc for
`getTargetQ()` gives no detail on whether this field is a raw echo of
the last `servoJ()` write, or an already-smoothed/interpolated internal
setpoint (the robot-side `servo_thread`'s own `get_target_waypoint()` +
`lookahead_time` blending, per item 4, strongly suggests the latter —
it would be the *output* of that blending, not the raw input). Without
reading the UR controller firmware's own source (not available here),
this cannot be verified either way from this checkout; it must not be
treated as a faithful echo of the exact commanded value without further
check (e.g., commanding a known step and confirming `getTargetQ()`
reproduces it exactly and immediately, which has not been done).
`getSpeedScaling()`/`getSpeedScalingCombined()` logging would close the
separate item-5 gap. None of this is proposed as something to build or
run now — this section states the gap, not a plan to close it.

## Hash re-verification (post-pass, unchanged)

```
0cb859a7bef44c3b060c34064263843e4674007de23ca9822c1d00e25cf03d27  sustained_worker_test_results_RECOVERED_ORIGINAL_23of90.json
9a13bf127c68221857bd1ba2f9e342c7edd902df783406b175280f9d48480b1f  sustained_worker_test_results_threadcompare_labmachine_20261116T173928Z_current.json
982a0032c9c73db738401d144c0dcaacae6fc5e798e79715af8d665ae512534b  sustained_worker_test_results_threadcompare_labmachine_20261116T173928Z_verified1thread.json
7197f0c5db1533bc1b323d09c3e06c6a90ecf2ac1a11de20768d508772c6184c  k2_post_tcpfix_threadfix_20261117T101102Z_20261117T101121Z/path_follow.jsonl
6aafeb66013bd72ec5fd55a5d544edbd58aa78e8db16c99fee68e243658d3398  k2_post_tcpfix_threadfix_20261117T101102Z_20261117T101121Z/summary.json
```

All five match the values already on record. `delay_samples=2`
unchanged everywhere. No controller, protocol, or hardware action was
taken; this is read-only investigation only. Day-1 stays blocked.
