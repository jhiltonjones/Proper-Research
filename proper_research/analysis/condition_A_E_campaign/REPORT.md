# Conditions A–E hardware campaign: read-only analysis

2026-11-20. Rectangle plan (`rectangle_10x15mm_skipglobal_L30to50mm_2026-11-19_noDipole`),
insertion envelope 20–70mm, measured-clearance-abort 230mm, in-QP magnet-exclusion
radius 233mm (230+3mm margin) for every MPC condition. No hardware was executed,
tuned, or modified to produce this report; all figures below are computed from
already-saved logs under `close_loop_logs/`.

Scripts, manifest, per-run/aggregate CSVs and plots live alongside this file in
`proper_research/analysis/condition_A_E_campaign/`.

## 1. Runs found and verified

23 candidate run directories exist under `close_loop_logs/` matching A–E naming.
**21 are included; 2 are excluded.**

| Condition | Candidates | Included | Excluded | Notes |
|---|---|---|---|---|
| A | 6 | 4 | 2 | excluded runs crashed before/during startup with zero `path_follow.jsonl` rows — these are the two documented pre-fix launch attempts (`baseline_A_source_manifest_2026-11-20.txt` v1/v2: `RobotJointStream.latest_joints_with_timestamp()` called `LatestSlot.get_with_timestamp()`, which did not exist until the v3 fix) |
| B | 5 | 5 | 0 | 4 completed (`path_complete`), 1 aborted (`tracking_error_exceeded(7.02mm > 7.00mm)` at tick 117/235, 54.7% of path). Reported separately per the user's instruction not to fold aborted runs into a "completed" aggregate, and not to hide the difficult part of the path by averaging over a shorter completed run. |
| C | 4 | 4 | 0 | no ambiguity |
| D | 4 | 4 | 0 | no ambiguity |
| E | 4 | 4 | 0 | no ambiguity |

**Condition wiring verified, not assumed:**
- **B (frozen Jacobian):** `controller_metadata.json`'s `schedule_cache_path`/`schedule_cache_sha256`
  point at the `_frozen_jacobian_schedule.npy` file (not the original scheduled-LTV
  file), and that file's own rows were independently loaded and confirmed identical
  across all 214 reference samples, for every one of the 5 B runs.
- **E (adaptation active):** `jacobian_adaptation_diagnostics.mode == "active"` on
  every logged tick across all 4 E runs, with real, varying accept/reject activity
  (103–115 of ~234 ticks accepted per run) — not a flag that was set but never
  exercised, and not all-rejected either.
- **C (reactive, no predictive geometric QP):** confirmed by reading
  `run_inv_7dof_delay_aware.py` directly — it has no in-QP geometric constraint by
  construction (there is no QP). The shared external measured-clearance check
  (`--measured-clearance-abort-mm`, 230mm, same constant as A/B/D/E) is wired and
  populated on every row of C's `path_follow.jsonl`. C has no
  `predicted_beam_positions.jsonl` at all, which is correct and expected — it has no
  prediction horizon to log.

**Source-version note:** per `docs/campaign_prep/baseline_A_source_manifest_2026-11-20.txt`,
the 4 included A runs, all 5 B runs, and all C/D/E runs were launched after v3 of the
source manifest (the `LatestSlot.get_with_timestamp()` fix) and after each condition's
own enabling fix (v4 for A's plotting-only fixes, v5 for B's `--freeze-jacobian`, v6
for C + the ported `--protective-config-reviewed` gate, v7 for D/E) — i.e. every
included run ran under the final, fixed source for its own condition.

## 2. Data quality and alignment

- `path_follow.jsonl` / `predicted_beam_positions.jsonl` join on `step` as an exact
  set match for every A/B/D/E run (the session's own `steps += 1` relocation fix,
  already regression-tested, holds on real hardware data).
- `deadline_miss` (per-tick, whole-tick) and `solver_deadline_missed` (solver-only)
  are logged as distinct fields; **both are zero on every tick of every included
  run in this campaign** — no deadline misses occurred at all, on any condition.
- `tip_mm` is the real measured-tip field (millimetres); earlier in this session a
  separate plotting script bug read a non-existent `tip_meas_m` field — not used
  here, confirmed the correct field was read throughout this analysis.
- Reference-sample count (214) is distinct from control-tick count (234–235);
  terminal hold is flagged separately (`terminal_hold`) and excluded from "active
  motion" tracking-error statistics below. Final error is reported separately from
  active-motion RMSE, never conflated.
- No missing value was treated as zero anywhere in this analysis; absent fields are
  reported as absent (e.g. C's missing `predicted_beam_positions.jsonl`, by design).

## 3. Per-run results (condensed; full 21-row table in `per_run_metrics.csv`)

All values: active-motion planar tracking error unless stated. Clearance is the
measured magnet-to-beam-base gap against the 233mm in-QP / 230mm external
threshold (A/B/D/E) or the 230mm external threshold only (C, no in-QP check exists).

| Condition | n | RMSE mm (mean±sd) | Final error mm (mean±sd) | Min clearance mm (mean) | Completion outcome |
|---|---|---|---|---|---|
| A | 4 attempts | 0.660 ± 0.060 | 2.125 ± 0.135 | 273.8 | 4/4 attempts completed |
| **B — all 5 attempts** | **5 attempts** | — (mixed; see split rows) | — | — | **4/5 attempts completed (80%); 1/5 aborted** |
| B — completed attempts only | 4 of those 5 | 0.628 ± 0.060 | 1.849 ± 0.209 | 279.6 | (subset of the row above — tracking statistics computed over completed attempts only, explicitly excluding the abort) |
| B — aborted attempt only | 1 of those 5 | 1.823 | 6.751 | 280.0 | stopped at 54.7% of path |
| C | 4 attempts | 0.506 ± 0.029 | 1.554 ± 0.218 | 276.1 | 4/4 attempts completed |
| D | 4 attempts | 0.486 ± 0.027 | 1.624 ± 0.175 | 275.0 | 4/4 attempts completed |
| E | 4 attempts | 0.581 ± 0.044 | 1.661 ± 0.350 | 274.3 | 4/4 attempts completed |

**On B's completion rate:** B attempted 5 hardware runs and completed 4 of them
(80%), with 1 outright abort on a tracking-error threshold. This 4/5 completion
outcome — not "4/4" — is B's true completion record and is reported here
explicitly rather than only in a separate sub-table. The RMSE/final-error/
clearance statistics labeled "B — completed attempts only" are computed
exclusively over the 4 completed attempts, by explicit construction (not by
omission) — they describe tracking quality conditional on completion, and should
not be read as B's unconditional tracking performance, which must include the
one failure.

**Clearance context:** every included run's minimum measured clearance was
273–280mm — 40–47mm above the 233mm in-QP boundary and 43–50mm above the 230mm
external threshold, on every single run. The constraint was never close to
binding anywhere in this campaign.

**Timing:** zero solver-deadline and zero whole-tick-deadline misses on every
included run, all five conditions.

**Command smoothness** (RMS joint-rate-command magnitude, rad/s, at the real
~10Hz tick interval): A 0.0086, B 0.0090, C 0.0101, D 0.0103, E 0.0088 — C and D
command slightly larger steps on average than A/B/E; none are large in absolute
terms.

**B's one abort, examined on its own executed portion (not matched-progress
averaged into B's completion rate):** stopped at `tracking_error_exceeded
(7.02mm > 7.00mm)`, 117/235 ticks, 54.7% of the path. Its RMSE over the executed
portion (1.823mm) and final error (6.751mm, i.e. right at the abort threshold) are
both far worse than any other run in the campaign, B or otherwise — this is a
genuine, isolated failure, not hidden by truncating the average to the easy part
of the path.

## 4. H1 — MPC (A) vs reactive inverse-Jacobian (C)

Paired across the 4 real hardware blocks (see §7 for the block structure):

- **(a) Which tracks better? C, consistently.** A−C active RMSE was positive in
  **all 4 blocks** (diffs: 0.247, 0.149, 0.153, 0.067mm; mean +0.154mm, sd 0.074,
  paired Cohen's d = 2.09). A−C final error was mixed in sign (−0.286, +0.133,
  −0.198, −0.687mm; mean −0.260mm) — A's RMSE is reliably worse than C's, but final
  error is less consistent.
- **(b) Which completes more without intervention? Tied, 4/4 both.**
- **(c) Is there direct evidence of a predictive-constraint benefit? No.** The
  in-QP magnet-exclusion constraint never approached activity in any A run (margin
  ≥40mm throughout). No offline same-state QP re-solve with the constraint removed
  was meaningful to run, because the constraint was never close enough to binding
  for its removal to plausibly change the chosen command — doing so would not be
  informative mechanistic evidence, only a confirmation that an inactive
  constraint changes nothing. **This dataset cannot support or refute H1's
  predictive-constraint claim**, because no run in the campaign ever tested it.

**H1 verdict, stated as two separate claims rather than one bundled verdict:**
- **On observed tracking:** A vs C favors INV — C tracked consistently better
  than A across all 4 paired blocks. This is a real, observed result.
- **On the predictive-constraint claim specifically:** this result does **not**
  contradict H1's predictive-constraint component. It does not test it at all.
  The in-QP magnet-exclusion constraint was never close to active in any A run
  (margin ≥40mm throughout), so C's better tracking cannot be attributed to —
  or taken as evidence against — the value of embedding geometric constraints in
  an optimizer. Absence of constraint activity means this claim is **untested**,
  not falsified.
- Completion was tied, 4/4 both.

Any A-vs-C tracking advantage found here is a
controller-package comparison (different prediction, objective, redundancy
handling) — it cannot be attributed specifically to embedding constraints in an
optimizer, because the optimizer's constraints were never exercised.

## 5. H2 — scheduled Jacobian (A) vs frozen Jacobian (B)

- **Hardware tracking/completion:** A−B active RMSE across the 4 paired completed
  blocks was small and mixed-sign (diffs: 0.134, 0.002, 0.071, −0.080mm; mean
  +0.032mm, sd 0.092, d=0.34) — **not a clean win for either**. A−B final error was
  consistently negative (A lower in all 4 blocks: −0.814, −0.033, −0.169, −0.142mm;
  mean −0.290mm, d=−0.82) — a more consistent finding, favoring A on final error
  specifically, separate from RMSE.
- **Causal prediction accuracy at matched lead times** (full-horizon forecasts
  saved before the corresponding future measurement arrived, not same-tick
  reconstruction): B was slightly *more* accurate than A at short lead times and A
  only pulled ahead at longer lead times (crossover around lead-stage j≈8); the
  gap at any lead stage was ≤0.02mm — a small, direction-reversing difference, not
  a clear advantage for the scheduled predictor. Full curve in
  `forecast_scheduled_vs_frozen.json` / plot3.
- **The one finding that does favor A:** B produced this campaign's only hardware
  abort (1 of 5 attempts, `tracking_error_exceeded`). A had zero aborts across 6
  attempts (2 pre-fix crashes excluded as unrelated to the Jacobian question, 4
  completions).

**H2 verdict: inconclusive on tracking RMSE (small, mixed-sign difference);
inconclusive-to-weakly-reversed on short-horizon forecast accuracy; the clearest
asymmetry is reliability — B had one run fail outright, A did not.** Increasing
`||J[k]−J[0]||` along the path was not, by itself, shown to cause the one B
failure; that failure is reported as an isolated event, not analyzed as a
dose-response relationship (insufficient n to do so credibly).

## 6. Secondary comparisons

**A vs D (N=15 vs N=3): contradicted, consistently.** A−D active RMSE was positive
in all 4 blocks (diffs: 0.210, 0.197, 0.155, 0.131mm; mean +0.173mm, sd 0.037,
d=4.75 — the most consistent effect in the whole campaign). A−D final error was
essentially flat (mean +0.009mm, mixed sign) and completion tied 4/4. With
delay_samples=2, N=3 gives the joint channel only one undelayed optimization
stage beyond the delay horizon, while the (always-undelayed) insertion channel
still gets all 3 stages — this asymmetry is real but this analysis did not find
it to explain worse performance for the *longer*-horizon A; if anything the
shorter-horizon D tracked more consistently. **This is the single most
interesting unexplained result in the campaign** and is flagged as a candidate
question for further offline (not hardware) investigation.

**A vs E (adaptation off vs on): inconclusive, weak signal toward E.** A−E active
RMSE was positive in 3 of 4 blocks and negative in one (diffs: 0.180, 0.100,
−0.018, 0.052mm; mean +0.078mm, sd 0.083, d=0.95). Final error was mixed (mean
−0.052mm). E's online estimator genuinely ran (see §1) but its accept/reject
pattern was not further correlated with tracking improvement in this analysis
given n=4; the `variance` diagnostic field is not treated as a calibrated
confidence value, per instruction — it is reported as a raw diagnostic only.

## 7. Statistical treatment and confounds

**The run is the independent unit; n=4 per condition** (n=5 for B including its
abort, reported separately). Pairing was NOT assigned by sorting or by assuming a
latent repetition number — it was derived from each run's own timestamp,
discovering that all five conditions were in fact run in four real, contiguous
**A→B→C→D→E hardware blocks** (verified from the actual run timestamps: each
block's five runs are minutes apart, in that fixed order, four times in sequence).
This is a genuine matched-block structure, so paired statistics (paired
differences, paired Cohen's d) are reported, not independent-sample statistics.

**Critical confound: condition order is perfectly confounded with position
within each block, across all four blocks** — A is always first, E is always
last, every time. Any systematic drift across a block (camera/robot warm-up, rig
settling, insertion-advancer wear, operator intervention between runs) would
produce exactly the same signature as a true condition effect, and this dataset
cannot distinguish the two. This is the single most important limitation on every
comparison above, including H1 and H2.

No p-values are reported. Cohen's d values above are descriptive effect-size
summaries of n=4 paired differences, not inferential test statistics; at n=4 they
carry very wide uncertainty and should not be read as settled effect magnitudes.
No multiplicity correction is applied or needed because no formal hypothesis test
was run. This is a pilot analysis of an opportunistic, order-confounded hardware
campaign, not a preregistered confirmatory study.

## 8. Deliverables

All in `proper_research/analysis/condition_A_E_campaign/` (canonical and
worktree, scripts identical by diff; data/plots canonical-only, as the
underlying logs only exist there):

- `build_manifest_and_metrics.py` → `manifest.json` / `manifest.csv` (23 runs, inclusion/exclusion with reasons)
- `verify_condition_wiring.py` (B/E/C wiring verification, printed evidence)
- `per_run_metrics.py` → `per_run_metrics.csv` / `.json` (21-row table, full §3 metric set per run)
- `condition_aggregates.py` → `condition_summary.csv` (per-condition mean/sd/median/range, B split completed-vs-aborted-vs-all-5)
- `forecast_scheduled_vs_frozen.py` → `forecast_scheduled_vs_frozen.json` (§5 lead-time forecast comparison)
- `statistics_section.py` → `paired_statistics.json` (§4/§6 paired diffs, sd, Cohen's d for A-C/A-B/A-D/A-E)
- `make_plots.py`, `plot_error_vs_progress.py` → 5 PNGs: tracking RMSE by condition, measured/commanded clearance vs threshold, scheduled-vs-frozen forecast error by lead time, E's adaptation gain/accept-reject trace, error vs normalized path progress (abort location retained for B)

### Evidence table

| Claim | Supporting runs/metrics | Limitations | Verdict |
|---|---|---|---|
| C tracks better than A (RMSE) | 4/4 paired blocks, d=2.09 | order-confound (C is 3rd of 5 in every block); controller-package, not isolated-factor, comparison | **Supported** (observationally, within this confound) |
| MPC's predictive constraints give a demonstrated adherence/completion advantage over C | — | constraint never active in any run | **Not established** — untested, not contradicted |
| A and B differ in tracking RMSE | 4/4 paired blocks, small mixed-sign diffs, d=0.34 | — | **Inconclusive** |
| A has lower final error than B | 4/4 paired blocks, all negative, d=−0.82 | n=4; order confound | **Supported**, final-error-specific |
| Scheduled Jacobian forecasts more accurately than frozen at short lead times | direct lead-time comparison, pre-measurement forecasts | gap ≤0.02mm, direction reverses at long lead times | **Contradicted at short lead, weakly supported at long lead** — net: not a clean win |
| B is less reliable than A | B: 1 abort / 5 attempts; A: 0 aborts / 4 post-fix attempts | n too small to estimate a failure rate precisely | **Supported**, single-event evidence |
| D (N=3) tracks better than A (N=15) | 4/4 paired blocks, d=4.75, most consistent effect in campaign | mechanism not identified; order confound (D is 4th of 5 every time) | **Supported observationally; mechanism not established** |
| E (adaptation on) tracks better than A (off) | 3/4 paired blocks positive, d=0.95 | 1 of 4 blocks reversed; small n | **Weakly supported, inconclusive** |

### Proposed paper wording

- **Better observed tracking:** "Under this campaign's order-confounded A→B→C→D→E
  hardware blocks (n=4), the reactive inverse-Jacobian controller (C) and the
  short-horizon MPC variant (D) both showed lower active-motion tracking RMSE than
  the scheduled-Jacobian N=15 baseline (A), consistently across all four blocks
  (paired d=2.09 and 4.75 respectively)."
- **Better observed prediction:** "The scheduled and frozen Jacobian predictors
  showed no clear overall advantage for either at matched lead times in this
  campaign; the frozen predictor was marginally more accurate at short lead times
  and the scheduled predictor only pulled ahead beyond roughly 8 control steps,
  with differences under 0.02mm throughout."
- **Demonstrated predictive-constraint intervention:** "No run in this campaign
  brought the magnet-exclusion constraint within 40mm of its boundary; these data
  therefore demonstrate neither a benefit nor a limitation of embedding this
  constraint in the optimizer, and any tracking-based comparison between MPC and
  the reactive controller should not be read as evidence about constraint
  handling specifically."
- **Observed completion/constraint adherence:** "All 21 included runs across
  conditions A, C, D and E, and 4 of 5 condition-B runs, completed the full
  rectangle path with zero deadline misses and clearance margins of at least
  40mm; one condition-B run aborted on a tracking-error threshold at 55% path
  completion."
- **Properties not established by these runs:** controller-order effects are
  confounded with condition throughout (A always first, E always last, in every
  block); the mechanism behind D's and C's more consistent tracking relative to A
  is not identified; the frozen Jacobian's one hardware abort is a single event,
  not a characterized failure rate; the online adaptation estimator's variance
  diagnostic is not a calibrated confidence measure.

**No further hardware is recommended by this analysis.** The one clearly
data-motivated open question — why both C (reactive) and D (short-horizon MPC)
tracked more consistently than A (long-horizon MPC) in a regime where no
constraint was ever active — is answerable, if at all, by further offline
analysis of the already-saved logs (e.g. examining A's own horizon-15 predicted
trajectories for systematic lag or overcorrection patterns absent from D's
horizon-3 predictions) before any new hardware run is considered.
