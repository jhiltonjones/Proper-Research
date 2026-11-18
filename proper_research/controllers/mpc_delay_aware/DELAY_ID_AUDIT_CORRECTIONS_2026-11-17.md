# Corrections to the 2026-11-17 delay-ID audit follow-up

**This document authorizes nothing.** No new hardware experiment, no
criterion approval, no controller/protocol/code change is authorized by
this document or by `DELAY_ID_AUDIT_FOLLOWUP_2026-11-17.md`. This is a
corrected evidence summary only. No hardware was run, no code or
protocol was changed, no git command was issued. `delay_samples=2` is
unchanged. Day-1 stays blocked. All protected result files are
unchanged (hashes re-verified at the end, matching the values already
on record).

## 1. Pickup latency was computed wrong — corrected

The prior report's 45.1ms mean / 85.9ms p99 for `pickup_latency_s`
included repeated pickups of an *unchanged* target across the
streamer's 20ms refresh cycle, not genuine new-command handoffs.
Recomputed directly from `arm_streamer_events.jsonl` (same file,
`sha256` unchanged — see end), filtering to `is_new_target == true`:

| filter | n | mean | p50 | p90 | p95 | p99 | max | min |
|---|---|---|---|---|---|---|---|---|
| `is_new_target=true` (genuine new handoff) | 250 | **5.132ms** | 5.303ms | 5.820ms | 5.907ms | **6.001ms** | 6.048ms | 1.870ms |
| `is_new_target=false` (repeated re-assertion) | 1000 | 55.132ms | 53.976ms | 85.389ms | 85.613ms | 85.938ms | 86.047ms | 21.874ms |

This independently reproduces the user's stated mean 5.13ms / p99
6.00ms over 250 new targets — confirmed, not just accepted. **The
45.1ms/85.9ms figures from the prior report must not be read as
new-command handoff delay.** They are the pickup-latency distribution
*including* repeated re-assertions of stale targets (4 reassertions per
genuine new target, matching the 20ms-refresh / ~100ms-tick ratio) and
describe something else (streamer refresh-loop behavior), not the
actuation-relevant handoff latency.

## 2. Perturbation amplitude — units/limit error, explicit withdrawal

The prior design proposed ±0.0025 rad/tick and, in the same sentence,
mislabeled it a "±0.25 rad/s velocity toggle." That is wrong by 10x:
0.0025 rad over a 100ms tick is **0.025 rad/s**, not 0.25 rad/s.

Worse, independent of that mislabeling: a two-level PRBS that reverses
sign in one 100ms tick (+0.025 rad/s → -0.025 rad/s) requires an
acceleration of (0.05 rad/s change) / (0.1s) = **0.5 rad/s²**, which
**exceeds** `JOINT_ACCELERATION_LIMIT_RAD_S2 = 0.40 rad/s²` — before
any nominal path motion is added on top of it.

**I explicitly withdraw** both of the prior document's claims that this
perturbation:
- (a) "remains safe when added to a worst-case nominal command", and
- (b) "necessarily produces sub-millimetre TCP motion."

Neither claim survives this correction. A reversal of this amplitude in
one tick is not safe at all under the project's own stated
acceleration limit, and no TCP-motion bound was actually derived — it
was asserted. Any future amplitude/switching-rate choice for this kind
of signal would need to be re-derived against the *acceleration* limit
directly (not just the per-tick position-step limit), and against a
real forward-kinematics TCP-displacement calculation, not an assertion.

## 3. Constraint protection for a combined (nominal + perturbation) command — design-only, not implemented

The prior design added the perturbation **after** the MPC's constrained
solve and claimed existing constraint protection held. That does not
follow: a post-hoc additive perturbation is applied outside the
optimization that enforces the TCP box, joint velocity/acceleration
limits, and the insertion-rate limit, so nothing guarantees the summed
command satisfies any of them. This section specifies, as design only
— **nothing here is built or run**:

- **Where the perturbation must live**: inside the QP, not downstream
  of it. Concretely, the perturbation must enter as an additional
  per-stage reference offset or cost term on the existing
  `input_tracking_weight`/reference trajectory (the same mechanism the
  controller already uses to track the nominal path's reference), so
  the solver's own box/velocity/acceleration/TCP-workspace constraint
  rows apply to the **combined** commanded trajectory the QP actually
  returns — not to the nominal path alone with an unconstrained
  perturbation bolted on afterward. Equivalently, the perturbation could
  be added to the *reference* `q_target`/`q_target_delta` fed into the
  existing MPC before it solves, so the solver's existing constraint
  set (already proven correct by `tcp_workspace_ax_refresh_self_test.py`
  for the TCP rows) is what ultimately clips it — never a term that
  bypasses the solve.
- **Cumulative/drift excursion**: bounding each tick's step does not
  bound the integral. A persistent PRBS bias (a real PRBS is zero-mean
  only over its full period, not over short spans) can accumulate
  genuine drift away from the nominal path across the excited segment.
  Any real design would need an explicit cap on cumulative joint
  displacement relative to the nominal path over the whole excited
  window (e.g. a drift budget enforced as its own constraint or
  monitored and aborted on), verified against actual accumulated
  `q_target_delta` sums over a candidate PRBS sequence before any
  hardware exposure — not inferred from the per-tick amplitude alone.
- **Transition acceleration**: the moment the perturbation switches on
  or off is itself a step discontinuity in the *reference*, which, once
  correctly routed through the QP (per the first point above), would be
  subject to the solver's own acceleration-limit rows at that tick —
  but this still needs an explicit check that the transition tick's
  required acceleration (nominal-path acceleration plus the onset/
  offset step) stays under 0.40 rad/s², the same arithmetic error from
  section 2 applied to the transition instants specifically, not just
  steady-state switching.

No PRBS signal, amplitude, or switching schedule is proposed as
final here — section 2's correction already invalidates the prior
amplitude, and this section's point is that a correct amplitude cannot
be chosen without first fixing where in the pipeline it is injected.

## 4. G_2/RMSE re-verification against hashed input, and a corrected (block) bootstrap

Re-ran the full 5-fold table directly against
`path_follow.jsonl` with `sha256 7197f0c5db1533bc1b323d09c3e06c6a90ecf2ac1a11de20768d508772c6184c`
(matches the value on record — same file):

| fold | held-out RMSE d=0 | d=1 | d=2 | d=3 | best d |
|---|---|---|---|---|---|
| 1 | 0.000519 | 0.000318 | 0.000254 | 0.000361 | 2 |
| 2 | 0.000177 | 0.000125 | 0.000131 | 0.000119 | 3 |
| 3 | 0.000377 | 0.000290 | 0.000288 | 0.000303 | 2 |
| 4 | 0.000494 | 0.000352 | 0.000395 | 0.000452 | 1 |
| 5 | 0.000512 | 0.000371 | 0.000329 | 0.000351 | 2 |

Vote counts: **{d=2: 3, d=1: 1, d=3: 1}** — this is the actual overall
winner (3/5, not a majority of 4/5). G_2 per fold: `[0.7606, 0.4486,
0.4150, 0.3622, 0.5865]` — identical to the numbers already on record
from the prior round; this re-derivation against the hash-confirmed
file found **no discrepancy** in the previously reported G_2/RMSE
values themselves. (G_1: `[0.6250, 0.5044, 0.4094, 0.4913, 0.4762]`;
G_3: `[0.5149, 0.5482, 0.3535, 0.1626, 0.5300]`.)

The prior round's bootstrap resampled individual ticks i.i.d., which
understates uncertainty for autocorrelated control-loop residuals.
Redone with a **moving block bootstrap** (block length L=10 ticks,
20,000 resamples), comparing the actual overall winner (d=2, 3/5 votes)
against both alternatives, pooled across all 5 folds' out-of-sample
residuals in original time order:

| comparison | point (RMSE_other − RMSE_d2) | 95% block-bootstrap CI | frac. resamples with opposite sign |
|---|---|---|---|
| d=1 vs d=2 | +0.000011 | [-0.000028, +0.000039] | 0.367 |
| d=3 vs d=2 | +0.000042 | [+0.000008, +00.000073] | 0.004 |

Corrected conclusion: **d=2 is not distinguishable from d=1** even
under the autocorrelation-respecting block bootstrap (CI straddles
zero, sign flips in 37% of resamples) — the "coin flip" framing was
closer to right for this specific comparison than the i.i.d. bootstrap
suggested. But **d=2 is clearly and robustly better than d=3** globally
(CI excludes zero, only 0.4% of resamples flip the sign) — this is a
real, non-marginal result the prior round did not report because it
only examined the two folds it had flagged as near-ties, not the full
winner-vs-alternatives comparison. Net effect on the overall claim: the
data support ruling out d=3, but do not support confidently separating
d=2 from d=1 — "d=2 confirmed" is not supportable either way on this
log; "d=2 vs d=1 is the live ambiguity, d=3 is excluded" is the
corrected, defensible statement.

## 5. Measurement scope — corrected with code evidence, not asserted

The prior report asserted "joint-encoder-only, not the camera chain"
without tracing it. Traced now through
`proper_research/hardware/online/close_loop_path_follow.py` (lines
1275–1296) and `camera_source.py` (lines 379–396):

- The joint *value* in `q_meas_rad` comes from
  `reader.latest_joints_with_timestamp(...)` — `reader` is a
  `RobotJointStream` (`state_stream.py`), the robot's own independently
  running RTDE joint-state stream. It is **not derived from camera
  pixels**; that part of the prior claim holds.
- But that getter is only *called* from inside `CameraSource`'s own
  per-frame processing path (`camera_source.py:381-389`, invoked
  alongside `vision_result`, `robot_pose_getter`, and
  `insertion_length_getter` in the same frame-processing step). The
  2026-11-16 comment at `close_loop_path_follow.py:1288-1291`
  states this getter exists specifically "to guarantee q_meas_rad and
  its own age are the SAME underlying sample" as the vision estimate —
  i.e. the joint sample is deliberately picked up **synchronized to the
  camera's frame-processing cadence**, not on an independent,
  continuously-running joint-sampling clock of its own.

**Corrected claim**: the logged joint samples are genuine,
independently-acquired robot-encoder/RTDE readings (not image-derived),
but their *acquisition timing* is coupled to the camera/vision
processing loop, not free-running. This is consistent with — and
explains — item 2's finding from the prior round that
`joint_sample_age_at_use_s` tracks `camera_vision_age_s` with a
near-constant ~10-11ms offset at every percentile: both ages are driven
by the same per-frame pickup event. The delay-identification script's
input data is therefore **not cleanly isolated from camera-loop
timing** the way "encoder-only" implied; any future identification
experiment that wants genuinely camera-independent joint timing would
need a getter that samples `reader` on its own clock, decoupled from
`CameraSource`'s frame cadence — which does not exist in the current
pipeline and is not proposed here.

## Hash re-verification (post-pass, unchanged)

```
0cb859a7bef44c3b060c34064263843e4674007de23ca9822c1d00e25cf03d27  sustained_worker_test_results_RECOVERED_ORIGINAL_23of90.json
9a13bf127c68221857bd1ba2f9e342c7edd902df783406b175280f9d48480b1f  sustained_worker_test_results_threadcompare_labmachine_20261116T173928Z_current.json
982a0032c9c73db738401d144c0dcaacae6fc5e798e79715af8d665ae512534b  sustained_worker_test_results_threadcompare_labmachine_20261116T173928Z_verified1thread.json
7197f0c5db1533bc1b323d09c3e06c6a90ecf2ac1a11de20768d508772c6184c  k2_post_tcpfix_threadfix_20261117T101102Z_20261117T101121Z/path_follow.jsonl
6aafeb66013bd72ec5fd55a5d544edbd58aa78e8db16c99fee68e243658d3398  k2_post_tcpfix_threadfix_20261117T101102Z_20261117T101121Z/summary.json
```

All five match the values already on record. `delay_samples=2`
unchanged everywhere. No controller, protocol, plan, or hardware
action was taken in producing this correction. **Nothing in this
document or the one it corrects authorizes a new experiment, a
criterion, or a code change** — those all still require the user's own
explicit, separate approval.
