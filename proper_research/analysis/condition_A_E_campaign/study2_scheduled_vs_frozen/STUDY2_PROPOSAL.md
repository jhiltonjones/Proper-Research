# Study 2: scheduled vs frozen task Jacobian — offline decomposition and frozen proposal

No hardware was executed and no existing source file or log file was modified
to produce this document. All numbers below come from two new scripts reading
already-saved logs: `decompose_jacobian_contribution.py` and
`fixed_history_replay.py`, both in this directory, with raw output in
`jacobian_contribution_term.json` and `fixed_history_replay.json`.

## 1. Mechanism, confirmed from source (not assumed)

Read directly from `delay_aware_mpc.py`'s `solve_delay_aware()`:

```
predicted_beam_position[k] = nominal_ref(idx_k) + J[k] @ (z_pred[k] - z_ref[k]) + bias[k]
```

- `nominal_ref` comes from `nominal_reference_positions_m`, which `worker_process.py`
  sets from the plan's own `desired_position_m` array — **identical for A and B**,
  never touched by `--freeze-jacobian`.
- `idx_k` for horizon stage `k` at a tick with `ref_index` is
  `clip(ref_index + 1 + k, 0, S-1)` (confirmed in
  `simulate_time_parameterized_beam_output_mpc.py`'s `_reference_indices()`).
  This is how `J_sched[k]` and `z_ref[k]` were reconstructed here — the logged
  `jacobian_used` field is **only stage 0's Jacobian**, not the full horizon
  window, so the window had to be rebuilt from the raw schedule file.
- `J0` (B's frozen Jacobian) was independently confirmed to equal
  `schedule[0]` of A's own schedule file, by loading B's actually-used frozen
  `.npy` file and checking every one of its 214 rows is identical to that value.
- `bias[k]` is a same-tick output-bias correction built from a running,
  low-pass-filtered measured-vs-nominal residual (`_estimate_output_residual`).
  **It is not independently recoverable from these logs** — it depends on the
  controller's own internal filter state, which is not saved per tick. This is
  stated as a genuine limitation, not approximated.

This confirms the three effects the brief asked to separate:
1. **Nominal model** (`nominal_ref`) — identical for A and B, cancels in any
   A-vs-B comparison.
2. **Output-bias correction** (`bias[k]`) — present in both, not independently
   recoverable from logs, also identical in structure (same filter, same
   inputs) for A and B.
3. **The Jacobian-choice term**, `(J_sched[k]-J0) @ (z_pred[k]-z_ref[k])` — the
   ONLY structural difference between what A's and B's forecasts would be for
   the same executed history. This is exactly the quantity the brief's formula
   asks for, and it is the only one of the three terms this analysis computes
   directly.

## 2. The Jacobian-choice term, by horizon stage and axis

Computed over all 939 successful ticks across the 4 completed condition-A runs
(`decompose_jacobian_contribution.py`). RMS magnitude of
`(J_sched[k]-J0) @ (z_pred[k]-z_ref[k])`, selected stages:

| Stage k | x RMS (mm) | y RMS (mm) | z RMS (mm) |
|---|---|---|---|
| 0  | 0.271 | 0.623 | 0.105 |
| 2  | 0.278 | 0.638 | 0.106 |
| 5  | 0.293 | 0.656 | 0.105 |
| 10 | 0.314 | 0.661 | 0.101 |
| 14 | 0.329 | 0.664 | 0.097 |

The term is **not negligible**: it grows modestly with lead stage in x and y,
stays roughly flat (slowly shrinking) in z, and the y-axis component (0.6mm
RMS) is an order of magnitude larger than A's own cross-run tracking RMSE
standard deviation (0.060mm, from the main campaign's `condition_summary.csv`).
**This term, considered alone, is large relative to run-to-run tracking
noise.**

## 3. Fixed-history replay (prediction replay, NOT closed-loop evidence)

Per `fixed_history_replay.py`: using ONE completed A run's actual recorded
execution history (`baseline_A_pilot_2026-11-20_retry1_20261120T122825Z`,
235 ticks), both predictors' forecasts were computed from the SAME `z_pred[k]`
(A's own actually-executed, already-optimized states — no future information
used, no re-solved QP) — the frozen forecast is the linear substitution
`scheduled_forecast[k] - (J_sched[k]-J0)@(z_pred[k]-z_ref[k])`, which follows
directly from §1's confirmed decomposition. Both were then scored against this
SAME run's actual future measured tip at matched lead times, using the
identical matching rule (nearest later tick within 0.5·dt) the main campaign
analysis already used.

| Lead stage j | Scheduled mean error (mm) | Frozen mean error (mm) | Frozen − scheduled (mm) |
|---|---|---|---|
| 0  | 0.241 | 0.416 | +0.175 |
| 4  | 0.498 | 0.654 | +0.156 |
| 8  | 0.552 | 0.767 | +0.214 |
| 11 | 0.560 | 0.807 | +0.248 |
| 14 | 0.569 | 0.829 | +0.261 |

**On this single fixed history, the scheduled predictor is more accurate than
the frozen substitution at every lead stage, with the gap growing from
+0.175mm to +0.261mm across the horizon** — a clean, monotonic, one-directional
result.

**This directly contrasts with the main campaign's own across-run lead-time
comparison** (§5 of `REPORT.md`), which found B's real closed-loop predictions
slightly MORE accurate than A's at short lead times, crossing over only past
j≈8, with gaps ≤0.02mm. The two results are not contradictory once the
mechanism is understood: the across-run comparison compared two DIFFERENT
closed-loop trajectories (A's real feedback loop vs B's real feedback loop,
each reacting in real time to its own controller's commands), while this
replay holds the trajectory FIXED and only swaps which linear model scores it.
**The scheduled Jacobian is a measurably better LOCAL PREDICTOR of a given
fixed trajectory; this does not by itself establish it is a better
CONTROLLER, because closed-loop feedback can compensate for an open-loop
model's prediction error in ways this replay cannot capture.** This is stated
explicitly as the resolution of the apparent tension, not glossed over.

## 4. Comparison against the empirical noise floor

A's and B's own cross-run tracking RMSE standard deviations are both ≈0.060mm
(`condition_summary.csv`). Both the decomposed term (§2, up to ≈0.66mm RMS in
y) and the fixed-history replay gap (§3, 0.175–0.261mm) **exceed this noise
floor by roughly 3–10×**. **The scheduled-vs-frozen prediction difference, in
this decomposed/replayed sense, is distinguishable from run-to-run
repeatability noise** — even though the main campaign's closed-loop A-vs-B
RMSE difference (mean +0.032mm, sd 0.092mm) was NOT distinguishable from that
same noise floor. The apparent contradiction is the same one resolved in §3:
a real, noise-exceeding open-loop prediction difference does not necessarily
translate into a comparably large closed-loop tracking difference, because
the controller re-optimizes every tick against fresh measurements.

## 5. Alternative paths — not triggered

The brief asks for a predefined alternative-path screen "only if the present
path provides insufficient discrimination." §2–4 show the current rectangle
path's Jacobian variation **is** sufficient to produce a clear, noise-exceeding
signal in both the decomposition and the fixed-history replay. **This task is
therefore not triggered**; no alternative path is screened or proposed here.
(Separately, and not part of this decision: this session's plan library
already has triangle, S-curve and higher-remanence plans built with precomputed
contact/no-contact schedules, if a future, different question specifically
needs more aggressive curvature — noted for completeness, not recommended
here.)

## 5b. Bounded-check extension (second pass) — exact equations, dual-bias comparison, all 9 histories

This section responds to a follow-up instruction and supersedes §3-4's
headline framing where they conflict; §3-4 above are kept unmodified as the
original, single-run, first-pass finding. No hardware, no source/log
modification; new script `extended_replay_multirun.py`, output
`extended_replay_multirun.json`, both in this directory.

**Exact replay equations, confirmed from source (not assumed):**

*Registration* — applied once at startup (`worker_process.py`'s
`set_frame_transform`), not per tick:
```
des_R = des_planner @ R_fit.T + t_fit
controller.reference.desired_position_m <- des_R
controller.nominal_reference_positions_m <- des_R.copy()
```
After this call, `nominal_reference_positions_m` and
`reference.desired_position_m` are the *same* array for the rest of the
run. This campaign's own startup logs show `R_fit ≈ I`, `t_fit ≈ 0mm`
(near-identity), so registration's numeric contribution here is
negligible — but it is a real, named step, reported rather than omitted.

*Per-tick residual* (`delay_aware_mpc.py`'s `_estimate_output_residual`,
confirmed in `simulate_time_parameterized_beam_output_mpc.py`):
```
index = clip(control_index, 0, S-1)                 # == ref_index here
local_nominal(J) = nominal_ref[index] + J[index] @ (z_meas - ref_state[index])
instantaneous(J) = measured_beam_position - local_nominal(J)
filtered_t = alpha*filtered_{t-1} + (1-alpha)*instantaneous_t      (t>0)
filtered_0 = instantaneous_0                                       (init)
```
**`disturbance_filter_alpha`'s confirmed default, and this campaign's
actual unoverridden value (verified against `controller_metadata.json`,
no override present), is 0.0.** With `alpha=0`, `filtered_t ==
instantaneous_t` on *every* tick — the filter has no memory at all. This
corrects the prior pass's description of this as a "running, low-pass-
filtered residual"; in this specific deployment it is a per-tick
recomputation with no persisted state, which is exactly what makes a
from-scratch, stateless replay faithful here (no internal controller
state needs to be reconstructed or approximated).

*Per-horizon-stage prediction* (`solve_delay_aware` /
`_dynamic_qp_terms_exec`, confirmed):
```
idx_window[k] = clip(ref_index + 1 + k, 0, S-1),  k = 0..N-1
predicted_beam_position(J)[k] = nominal_ref[idx_window[k]]
    + J[idx_window[k]] @ (z_pred[k] - ref_state[idx_window[k]])
    + beta_d * residual_t(J)       # same value tiled across every k
```
`J` is `schedule[.]` for the scheduled predictor, `J0 = schedule[0]` for
the frozen predictor — substituted *everywhere* `J` appears above,
including inside the residual, which is exactly what separates (b) from
(a) below.

**(a) Fixed-bias substitution** (§3's original method, generalized to
both A's and B's own native predictor):
```
other_forecast_a[k] = own_logged_forecast[k]
    - (J_own[idx_window[k]] - J_other[idx_window[k]]) @ (z_pred[k]-ref_state[idx_window[k]])
```
Reuses the run's own actually-realized residual unchanged for the other
Jacobian's horizon term only — not a faithful stand-alone predictor.

**(b) Separate causal bias** — the other Jacobian's residual is
recomputed independently at the same tick, then its whole forecast is
built end-to-end with its own Jacobian throughout:
```
residual_other_t = measured_beam_position
    - (nominal_ref[ref_index] + J_other[ref_index] @ (z_meas-ref_state[ref_index]))
other_forecast_b[k] = nominal_ref[idx_window[k]]
    + J_other[idx_window[k]] @ (z_pred[k]-ref_state[idx_window[k]]) + beta_d*residual_other_t
```

**A labeling bug was found and fixed while building this extension**: an
A run's own logged forecast is scheduled-native (A's worker solves with
the real schedule); a B run's own logged forecast is **frozen**-native
(B's worker solves with J0 throughout, confirmed in §1/the main
manifest). A first draft of this script treated every run's logged
forecast as scheduled-native, silently mislabeling B's real data. Fixed
by recomputing *both* predictors from primitives (`z_meas`, `z_pred`,
`measured_beam_position_m`, `ref_index`) on every run, using the run's
own logged forecast only as a sanity check against its own native
reconstruction (confirmed to match to <5×10⁻¹² mm on all 9 histories —
the equations above are exact, not approximate).

**(a) and (b) agree almost exactly on this data** (max difference
~10⁻⁷mm at every lead stage, every run) — the fixed-bias simplification
in §3's original pass was not distorting that result; with
`alpha=0`, the two bias formulations are numerically very close here.

**Evaluated on all 9 available histories** (4 completed A, 4 completed B,
1 aborted B — scoring window for the abort explicitly ticks 0–116 only,
the executed portion before the `tracking_error_exceeded` stop at tick
117; later ticks are excluded, not scored, not treated as zero).
Future-time alignment verified per tick/lead-stage before scoring, reusing
`plot_pilot_run.py`'s own `max_gap_s = 0.5·dt` matching rule; denominators
reported per run (105 of 235×15 tick-lead pairs excluded per full run for
lacking a future measurement within the gap tolerance near the run's own
end — this is identical and symmetric across all 9 histories, confirming
no selective exclusion).

**The result reverses depending on whose history is replayed — this is
the single most important finding of this extension:**

| | lead 0 (mm) | lead 8 (mm) | lead 14 (mm) |
|---|---|---|---|
| A-native runs (n=4), frozen−scheduled | −0.04 to −0.05 | −0.01 to +0.04 | **+0.04 to +0.09** |
| B-native completed runs (n=4), frozen−scheduled | −0.04 | −0.10 to −0.11 | **−0.16 to −0.18** |
| B-native aborted run (n=1, ticks 0–116), frozen−scheduled | −0.47 | −0.27 | −0.14 |

(Negative = frozen more accurate; positive = scheduled more accurate.)

- At **short lead times (j=0), frozen is more accurate on every one of
  the 9 histories**, A's and B's alike — consistent with the main
  campaign's own across-run finding (§5 of `REPORT.md`).
- At **long lead times, the result diverges by which history is
  replayed**: on A's own 4 histories, scheduled pulls ahead (crossover
  near j≈8, matching the original single-run pass); on B's own 5
  histories (including the abort), frozen's advantage *grows* with lead
  time instead of shrinking.
- **Per-run consistency within each native-group is high** (A's 4 values
  at lead 14: +0.094, +0.044, +0.058, +0.060, SD≈0.02mm; B's 4 completed
  values: −0.162, −0.176, −0.166, −0.162, SD≈0.006mm) — this is not noisy
  or inconsistent within a group, only opposite *between* groups.

**Uncertainty characterization, corrected per instruction:** the spread
reported above (SD≈0.006–0.02mm within each native-group of 4 runs) is
the genuine repeated-measurement spread of *this replay computation*
across independent hardware executions of the same nominal path under
the same controller. **This is not the main campaign's tracking-RMSE
standard deviation, and that statistic is not used here as a noise
floor** — the two answer different questions (closed-loop tracking
repeatability vs. replay-computation repeatability) and must not be
conflated, per instruction. The within-group spread above is small
relative to the +0.04 to −0.18mm effect sizes it is being compared
against, supporting that the directional result is not simply noise —
but with only 4-5 runs per group this is a descriptive observation, not
a calibrated statistical claim.

**On "closed-loop feedback absorbs the advantage" — replaced:** the prior
single-run framing ("the scheduled Jacobian is a measurably better local
predictor; closed-loop feedback evidently absorbs most of that advantage
before it shows up in net tracking") is **not supported by this
extension and is withdrawn as stated.** The extension shows the "better
predictor" result is not a fixed property of the Jacobian choice at
all — it **reverses by which controller's own trajectory is replayed**,
growing in favor of whichever Jacobian that trajectory's own controller
actually used, as lead time increases. A plausible but **unconfirmed**
alternative interpretation: each controller's closed-loop corrective
behavior may shape a trajectory that happens to be easier for its own
native model to predict further ahead (a form of selection effect, not
feedback "absorbing" an advantage that exists independent of which
trajectory is examined). This analysis does not isolate that mechanism —
it would require, at minimum, directly comparing each controller's own
real-time prediction-residual growth against the other's hypothetical
residual growth on a matched trajectory, which is not available from
these logs. An equally viable alternative, already noted in the main
campaign report: ordinary run-to-run confounds (initial state,
registration, camera noise, the campaign's own A→B→C→D→E order confound)
could independently explain why the closed-loop comparison did not
reproduce either single-history replay's result. **Both explanations
remain open; neither is established by this analysis.**

## 6. Proposed frozen experiment

**Hypothesis (revised per §5b; the single-run "feedback absorbs the
advantage" framing is withdrawn):** §5b found that which predictor
(scheduled or frozen) replays more accurately on a fixed history depends
systematically on which controller actually generated that history, with
the native predictor's relative advantage *growing* with lead time. The
revised, testable hypothesis is: **a controller's own closed-loop
corrective behavior shapes a trajectory on which its own native Jacobian
choice predicts further ahead more accurately than the alternative
would** — i.e. the predictor-accuracy ordering found in §5b is a
consequence of which controller drove the trajectory, not a fixed,
trajectory-independent property of either Jacobian. This is explicitly
one candidate mechanism among at least two (the other: ordinary run-to-
run confounds already documented in the main campaign report) — §5b does
not isolate which, if either, is correct.

**Primary metrics:**
1. Active-motion tracking RMSE (closed-loop), as the main campaign measured it.
2. Per-run, per-lead-stage (frozen − scheduled) replay difference computed
   identically to §5b, on any new run's own logs, reported separately by
   which controller generated the history (not pooled) — the key
   diagnostic is whether the sign/magnitude pattern found in §5b (short-
   lead frozen advantage on every history; long-lead advantage flipping
   to favor whichever controller is native) reproduces on new data, not a
   single pooled "gap" number.
3. If feasible from new logs: each controller's own real-time prediction-
   residual growth rate, to directly test the "own-trajectory favors own
   model" mechanism against the run-to-run-confound alternative — this
   specific comparison is what §5b lacked and could not isolate.

**Offline evidence supporting this hypothesis:** §5b above, in full
(supersedes §2-4's single-run framing, which is retained unmodified as
the original first-pass finding, not deleted).

**Balanced/randomized execution order:** unlike the original A→B→C→D→E block
structure (condition always in the same position, a confound stated
throughout the main campaign report), any new data collection for this
specific question should randomize or balance the A/B run order within each
repetition block (e.g. alternate A-first/B-first across repetitions, or draw
the order by coin flip per block and record it), so that any order/drift
effect can be checked for separately from the Jacobian-choice effect, which
was not possible in the original campaign.

**This proposal is frozen pending review.** No new hardware run has been
executed to produce or validate it; all evidence above comes from already-
existing logs, read and analyzed offline only.
