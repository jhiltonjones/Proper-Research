# Delay-identification audit follow-up (2026-11-17)

**SUPERSEDED IN PART.** The user found 5 concrete errors in this
document (wrong pickup-latency number in section 2 — it conflated
repeated re-assertions with genuine new-target handoff; a units/limit
error in section 4's perturbation design that makes the stated safety
claims false; an unjustified constraint-protection claim in section 4;
unverified G_2/RMSE numbers and an i.i.d.-only bootstrap in section 1;
an unverified "encoder-only" claim in section 5). See
`DELAY_ID_AUDIT_CORRECTIONS_2026-11-17.md` for the corrected numbers,
explicit withdrawals, and the real code-traced answer on measurement
scope. This file is kept unmodified below as the historical record of
what was originally claimed — do not cite its section-1/2/4/5 numbers
without reading the corrections file first.

Proposal/design document only. No hardware execution, no controller or
protocol changes (silent or explicit) in this pass. `delay_samples=2` is
unchanged everywhere. Day-1 stays blocked. All computations below were
re-run against the real logs in this session; none of the protected
result files were modified (hashes re-verified at the end).

## 1. Language correction: "statistically marginal" / "coin flip"

The prior round used that language for fold 2's 4.73% gap and fold 3's
0.48% gap (1-indexed; folds 2/3 == 0-indexed folds 1/2) without a formal
uncertainty analysis. That analysis is now done: paired bootstrap
(20,000 resamples, resampling held-out ticks with replacement) on the
per-tick squared residuals used by `delay_identification_5fold.py`
itself, comparing RMSE(d=1) and RMSE(d=2) within each fold (n=50
held-out ticks/fold on this log):

| fold (1-idx) | RMSE(d=1)-RMSE(d=2), point | 95% bootstrap CI | frac. of resamples with opposite sign |
|---|---|---|---|
| 2 | -7e-6 rad | [-4.2e-5, +2.8e-5] | 0.352 |
| 3 | +1e-6 rad | [-5.7e-5, +5.5e-5] | 0.462 |

Both CIs straddle zero by a wide margin, and the sign of "which delay
wins" flips in 35-46% of resamples. That is real evidence, not an
adjective. Going forward: replace "statistically marginal" / "coin
flip" with **"not statistically distinguishable at n=50 per fold
(bootstrap 95% CI for the RMSE(d=1)-RMSE(d=2) gap straddles zero; the
apparent winner's sign flips in 35-46% of resamples)"**. This also
means the previously reported `G_2` values for this run (0.41-0.55,
computed below) themselves carry wide, overlapping bootstrap CIs
(roughly ±0.25) — they should be reported with that CI attached, not as
point estimates, whenever this log is cited again.

## 2. Real sample-age / pickup-latency distributions (this run's own data)

Confirmed fields present in `path_follow.jsonl` (251 rows) and
`arm_streamer_events.jsonl` (2753 events, 1250 `pickup` events with
`pickup_latency_s`):

| quantity | n | mean | p50 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|
| `camera_vision_age_s` (== `state_age_s`) | 251 | 23.85ms | 20.50ms | 51.10ms | 57.75ms | 66.25ms | 69.80ms |
| `joint_sample_age_at_use_s` | 251 | 34.81ms | 30.40ms | 60.40ms | 66.50ms | 75.20ms | 79.90ms |
| `joint_sample_age_at_log_s` | 251 | 34.92ms | 30.50ms | 60.50ms | 66.60ms | 75.35ms | 80.00ms |
| streamer `pickup_latency_s` | 1250 | 45.13ms | 45.31ms | 85.31ms | 85.52ms | 85.92ms | 86.05ms |

Observations: the joint-sample age is consistently ~10-11ms older than
the camera-vision age at every percentile (both measured against the
same tick), and the streamer pickup latency is itself close to a full
100ms control period at the high end (p90-p99 cluster at 85-86ms) —
i.e. on a non-trivial fraction of ticks the picked-up state is nearly
one whole period stale by the time it's used.

What would distinguish the three candidate explanations (sample age /
pickup latency / genuine physical response dynamics) from each other,
concretely:
- **Sample age vs. physical dynamics**: re-run the identical 5-fold
  procedure restricted to the subset of ticks with `camera_vision_age_s`
  below its own median (freshest third) vs. above its own 90th
  percentile (stalest decile). If the identified delay (or the
  held-out RMSE gap) shifts with staleness, age is a live confound; if
  it doesn't move, age in this range isn't driving the result.
- **Pickup latency vs. physical dynamics**: the streamer's
  `pickup_latency_s` is a *different* clock than `camera_vision_age_s`
  (one measures queue/IPC delay to pick up a new state, the other
  measures how old the picked-up measurement is relative to when it was
  captured) — stratifying by `pickup_latency_s` the same way isolates
  whether IPC/queueing jitter (as opposed to sensor staleness) is doing
  the work.
- **Sample age vs. pickup latency vs. each other**: both are already
  present simultaneously in every tick of this log, so they cannot be
  separated from *this* data by construction — they are correlated with
  each other (both driven by the same background-streaming pipeline).
  Distinguishing them requires a run where one is deliberately varied
  independently of the other (e.g., forcing the streamer to always
  re-pick up before use, eliminating pickup-latency variance, while
  camera-vision age is left to vary naturally with scene processing
  time) — not something this pass proposes doing, just naming what
  observation would be needed.
- None of the above can separate "stale measurement" from "genuinely
  slow physical/actuation response" using only passive tracking data,
  because the controller's own commanded deltas are highly
  autocorrelated along the nominal path (low command excitation was
  already flagged in the prior round as the dominant information
  bottleneck). That is the direct motivation for item 4's experiment.

## 3. Threshold provenance: was `G_2>0.9` / `>=4/5` ever user-approved?

Traced through everything in this session's own record, including
`HOWTO_CAMPAIGN2.md`'s own text. Finding: **no**, not as a distinct,
explicit sign-off from the human user on that specific numeric bar.

What the record actually shows:
- The rule appears in `HOWTO_CAMPAIGN2.md` (lines ~433-435), justified
  there by a synthetic validation table (lines 578-588) run on
  simulated ticks with a *known* true delay — showing the rule
  correctly confirms d=2 when d_true=2 (G_2=1.00 on all 5 folds) and
  correctly stops when d_true=3 (G_2=0.84-0.89, below the 0.9 bar). That
  is a self-consistency check the rule was designed against, authored
  within this same investigation — not evidence the user specifically
  reviewed and approved "0.9" and "4/5" as the acceptance numbers.
- Line 446 records that this exact rule, applied to an earlier real run,
  already produced a STOP (best d=1 on all 5 folds, G_2 0.20-0.36) —
  so this is not the first time the rule has said STOP; it has never
  yet said CONFIRMED on real hardware data.
- I found no message in this conversation's history where the human
  user was asked to approve or was told "the bar is G_2>0.9 and 4/5
  votes, confirm?" and responded with approval.

Going forward: every historical STOP output already reported (the
prior real run's 0.20-0.36 G_2, and this run's 0.41-0.55) is preserved
exactly as computed and reported — those numbers don't change. But the
rule itself should be relabeled as **a self-derived methodological
default from an earlier round of this investigation, validated only
against synthetic known-delay data, not a criterion the user has
explicitly signed off on** — until the user does explicitly approve a
number (same 0.9/4-5, or a revised one).

## 4. Proposed diagnostic experiment (design only, not run)

**Goal**: increase command excitation so held-out RMSE at different `d`
actually separates, instead of relying on the nominal path's
autocorrelated commands.

**Signal**: PRBS (pseudo-random binary sequence), not a chirp/multi-sine,
because at a 100ms tick with a target lag of 2-3 samples, a PRBS's flat
power spectrum up to its switching-rate Nyquist gives even energy at the
1-3 sample lags being discriminated, and its binary two-level amplitude
makes the "did this joint actually move by Δ at lag d" comparison
simple to read by eye in addition to by RMSE — a multi-sine would need
phase unwrapping per frequency to get the same diagnostic read at only
3 candidate lags.

- **What it perturbs**: a small *additive* per-tick joint-velocity
  command perturbation `δq̇_j[k]` layered on top of the existing
  insertion-anchored MPC's nominal output on the already-approved
  marginfix plan — not a new path, not a relaxed constraint.
- **Amplitude**: bounded by `MAX_JOINT_STEP_RAD = 0.010 rad/tick`
  (`JOINT_VELOCITY_LIMIT_RAD_S = 0.10 rad/s` × `dt=0.1s`, both from
  `rectangle_stage_a/common.py`) and `JOINT_ACCELERATION_LIMIT_RAD_S2 =
  0.40 rad/s²`. Propose the perturbation amplitude at **25% of the
  per-tick step limit** (±0.0025 rad/tick, i.e. a two-level ±0.25
  rad/s velocity toggle), well inside both the velocity and
  acceleration limits even added on top of a worst-case nominal command,
  and inside the TCP box's 15mm QP-enforced robust margin
  (`--tcp-workspace-robust-margin-mm 15`, confirmed from this run's own
  `controller_metadata.json`) — a 0.0025 rad joint excursion maps to
  well under 1mm of TCP motion at this robot's link lengths, nowhere
  close to consuming that margin. This reuses existing limits; nothing
  is raised or relaxed.
- **Switching rate / frequency content**: PRBS register length and
  clock chosen so the sequence's dominant period covers 2-8 ticks
  (i.e. switches every 1-4 ticks), which puts spectral energy across
  the 1-3 sample lag band the 5-fold procedure is trying to
  discriminate (d∈{0,1,2,3}); a slower PRBS (switching every >10 ticks)
  would look locally like the already-tried smooth nominal path and
  reproduce the same low-excitation problem.
- **Duration**: bounded at the same ~250-tick (25s) scale as this run,
  which already gives 50 held-out ticks/fold — the excitation-dependence
  finding from the prior round means duration isn't the lever, command
  *variance* is; no justification exists yet for a longer run, so none
  is proposed.
- **Pose / constraints / abort**: identical start pose, same
  `--max-tracking-error-mm 7`, same TCP-workspace-in-QP box, same
  70ms deadline / 100ms period / `delay_samples=2`, same 5-consecutive-
  miss abort — no new or relaxed constraint of any kind.

## 5. What this experiment measures

**Joint-encoder response only**, not the camera/vision chain. This
matches `delay_identification_5fold.py`'s existing fields exactly
(`q_meas_rad`, `q_target_delta_rad` — both joint-space, read directly
off the robot's own joint encoders via the controller's state stream) —
the script never touches `camera_vision_age_s`/tip position. Command
timestamp = the tick `k` at which `q_target_delta_rad[k]` was computed
and sent; measurement timestamp = the joint encoder sample at tick
`k+1` that yields `realized_step[k] = q_meas_rad[k+1] - q_meas_rad[k]`.
This experiment is therefore scoped to *actuation* lag only; it cannot
by itself speak to vision-pipeline latency (item 2's `camera_vision_age_s`
distribution) — that would need a separate experiment correlating
commanded TCP perturbations against the vision-measured tip position,
not proposed here.

## 6. Pre-registered analysis plan (written before any data exists)

- **Structure**: same 5-fold contiguous-block leave-one-out structure as
  `delay_identification_5fold.py`, unmodified, run on the new PRBS log.
- **Near-tie / uncertainty handling (revised per item 1)**: for every
  fold, compute the same paired bootstrap (20,000 resamples) on
  RMSE(d)-RMSE(d') for the top-2 candidate delays by held-out RMSE.
  Define a **pre-registered tie rule**: if the 95% bootstrap CI of that
  gap straddles zero, the fold's vote is recorded as "tied" rather than
  assigned to a single `d` — ties are reported separately from clean
  votes in the final tally, not silently broken in favor of the lower
  RMSE.
- **Acceptance criteria**: propose *keeping* the same reporting
  quantities (`G_d`, per-fold vote) for comparability, but explicitly
  flagging that **any specific numeric bar (0.9 / 4-of-5, or a revision
  of it) is a new proposal requiring the user's own explicit approval
  before this experiment is treated as a Day-1 gate** — per item 3's
  finding that the existing bar was never explicitly approved either.
  No bar is silently carried over.
- **Consequence of either outcome**: if the PRBS run's bootstrap CIs
  separate cleanly (no overlap) and consistently favor d=2, that
  directly supports keeping `delay_samples=2` in the controller's
  prediction model and would strengthen the paper's delay-model claim
  with a real, non-marginal result instead of the current log's
  indistinguishable 0.41-0.55 `G_2` range. If it does *not* separate —
  or separates in favor of a different `d` — that is evidence the
  prediction model's delay term needs revisiting (or that this plant's
  true delay isn't a clean integer multiple of the 100ms tick, echoing
  the historical S10.2 fractional-model exploration), and the paper's
  current delay-model claim should be qualified accordingly rather than
  stated as confirmed. Either way, nothing about `delay_samples=2`'s
  current *value* changes until that experiment is actually run and
  analyzed — this document only proposes the design.

## Hash re-verification (post-pass, unchanged)

```
0cb859a7bef44c3b060c34064263843e4674007de23ca9822c1d00e25cf03d27  sustained_worker_test_results_RECOVERED_ORIGINAL_23of90.json
9a13bf127c68221857bd1ba2f9e342c7edd902df783406b175280f9d48480b1f  sustained_worker_test_results_threadcompare_labmachine_20261116T173928Z_current.json
982a0032c9c73db738401d144c0dcaacae6fc5e798e79715af8d665ae512534b  sustained_worker_test_results_threadcompare_labmachine_20261116T173928Z_verified1thread.json
7197f0c5db1533bc1b323d09c3e06c6a90ecf2ac1a11de20768d508772c6184c  k2_post_tcpfix_threadfix_20261117T101102Z_20261117T101121Z/path_follow.jsonl
6aafeb66013bd72ec5fd55a5d544edbd58aa78e8db16c99fee68e243658d3398  k2_post_tcpfix_threadfix_20261117T101102Z_20261117T101121Z/summary.json
```

All five match exactly. `delay_samples=2` unchanged everywhere. No
controller, protocol, or plan file was touched. Day-1 remains blocked
on the user's own explicit approval of whatever gate (existing or
revised) they choose after reviewing item 3.
