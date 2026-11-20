# Jacobian study: frozen hardware proposal

No hardware was executed to produce this document. This supersedes nothing
in `STUDY2_PROPOSAL.md` — that document's conclusions, including the
history-dependent reversal finding in §5b, are kept unmodified and remain
the authoritative record of the prior analysis passes. This document adds
one new, important correction (§1) and a concrete trial design (§2–5).

## 1. Verification of the alpha=0 bias-equivalence claim

### 1a. Algebraic derivation

From `_estimate_output_residual` (confirmed directly in
`simulate_time_parameterized_beam_output_mpc.py:879-904`):

```
index = clip(control_index, 0, S-1)
local_nominal(J) = nominal_ref[index] + J[index] @ (z_meas - ref_state[index])
instantaneous(J) = measured_beam_position - local_nominal(J)
filtered_t(J) = alpha*filtered_{t-1}(J) + (1-alpha)*instantaneous_t(J)      (t>0)
filtered_0(J) = instantaneous_0(J)
```

Two predictors (scheduled: `J=schedule[.]`; frozen: `J=J0=schedule[0]`)
differ in `instantaneous_t` only through

```
instantaneous_sched_t - instantaneous_frozen_t
    = (J0 - J_sched[index]) @ (z_meas_t - ref_state[index])
```

— the Jacobian-difference term evaluated at the **current** tick's own
tracking error, not at a future horizon offset. With `alpha=0`,
`filtered_t ≡ instantaneous_t`: the filter carries no memory, so each
predictor's bias state at any tick depends only on that tick's own
Jacobian-difference term, with no accumulation across ticks. This is the
correct, narrow scope of what `alpha=0` actually buys: **it rules out
cross-tick accumulation of the bias divergence, not the existence of a
per-tick divergence.** The per-tick divergence itself is governed by how
large `(J0-J_sched[index])@(z_meas-ref_state[index])` is at each tick —
empirically not negligible (§1c below), independent of alpha.

### 1b. A genuine bug found in the prior pass's "(a)≈(b)" claim

Re-implementing (a) and (b) independently (not reusing
`extended_replay_multirun.py`'s own helper functions) to cross-check,
this verification found that **the prior pass's "(a) and (b) agree almost
exactly (~10⁻⁷mm), a direct consequence of alpha=0" is not a real finding
about alpha — it is an algebraic tautology produced by how that script's
`other_fixed_bias` was constructed.**

That script computed, for an A run (own-native = scheduled):
```
other_fixed_bias = own_logged_forecast - (forecast_sched - forecast_frozen_b)
```
Since `own_logged_forecast ≡ forecast_sched` (confirmed by that same
script's own sanity check, matching to <5×10⁻¹² mm), this reduces by
direct substitution to:
```
other_fixed_bias = forecast_sched - forecast_sched + forecast_frozen_b
                  = forecast_frozen_b     (exactly, for any alpha)
```
**(a) was defined as an expression that cancels down to equal (b)
identically, by construction — not because the two represent genuinely
different bias-handling strategies that happen to coincide under
`alpha=0`.** The subtracted term `(forecast_sched - forecast_frozen_b)`
contains **both** the linear-term delta `(J_sched[idx]-J0)@dz` **and** the
bias-term delta `beta_d*(residual_sched-residual_frozen)` — whereas a
genuine "fixed-bias substitution" (per the original brief's intent: keep
the bias fixed, swap only the linear horizon term) must subtract **only**
the linear-term delta, leaving the native bias untouched. The prior
script did not do this, so no real comparison between (a) and (b) was
ever performed; the reported near-exact agreement was a tautology holding
for *any* alpha, including values the prior pass never actually tested.
This is exactly the kind of error the current brief's "verify
algebraically and with a nonzero-deviation test" instruction was designed
to catch.

### 1c. Corrected (a) vs (b), including the nonzero-alpha test

`alpha_verification_test.py` (this directory) re-implements both
predictors from primitives with a correct separation:
- **(b) separate causal bias**: a properly stateful recursive filter run
  in real tick order, independently for each predictor's own Jacobian
  (`filtered_sched_t`, `filtered_frozen_t`), each with its own memory.
- **(a) fixed-bias substitution, corrected**: take the native (scheduled,
  for an A run) predictor's own bias state and reuse it unchanged,
  subtracting **only** the linear term `(J_sched[idx]-J0)@dz` to score
  the frozen alternative — the bias itself is never touched.

Tested at lead stage 14, on `baseline_A_pilot_2026-11-20_retry1_20261120T122825Z`
(the same run as the original single-run replay), at `alpha ∈
{0.0, 0.1, 0.3, 0.5, 0.7, 0.9}` (0.0 reproduces the campaign's real,
unoverridden setting; the rest are artificially injected for this test
only, in a standalone script, never touching production source):

| alpha | mean |a−b| (mm) | max |a−b| (mm) | p95 |a−b| (mm) |
|---|---|---|---|
| 0.0 | 0.444 | 2.099 | 1.455 |
| 0.1 | 0.444 | 2.081 | 1.434 |
| 0.3 | 0.441 | 2.032 | 1.462 |
| 0.5 | 0.436 | 1.944 | 1.466 |
| 0.7 | 0.421 | 1.774 | 1.460 |
| 0.9 | 0.348 | 1.380 | 1.222 |

**Corrected conclusion: (a) and (b) do NOT agree closely, even at the
real `alpha=0` setting** — mean divergence ≈0.44mm at lead 14, not
~10⁻⁷mm. Divergence is present across the whole tested alpha range and
decreases mildly (not grows) as alpha increases, because the underlying
per-tick Jacobian-difference signal along this path is roughly
stationary (bounded by the controller's own tracking performance) rather
than accumulating; low-pass filtering a roughly-stationary signal
smooths its variance without amplifying its mean. **This means §5b's
reported "(a)≈(b)" agreement should be read as an artifact of that
script's own algebra, not as evidence that the fixed-bias and
separate-causal-bias framings are interchangeable under this campaign's
real alpha=0 setting.** §5b's *other* findings (the history-dependent
reversal across A-native vs B-native runs, computed via genuinely
reconstructed — not tautologically-collapsed — predictors in
`extended_replay_multirun.py`'s (b)-equivalent path, cross-checked
against each run's own logged native forecast) are not affected by this
correction, since that comparison never relied on the (a) construction
at all.

## 2. Candidate trajectory screening — not completed this pass

The brief asks to screen the existing 10×15mm rectangle plus 1.5×
(15×22.5mm) and 2× (20×30mm) scaled candidates for Jacobian-sensitivity
variation and feasibility. **No 1.5×/2× scaled rectangle plan at this
campaign's exact configuration (B_r=0.80T, no-dipole, skip-global)
exists yet** (confirmed: only `rectangle_10x15mm_2026-09-12`,
`..._skipglobal_2026-09-12`, `..._closed_Br080_2026-10-21`,
`..._closed_Br144_2026-10-22`, `..._L30to50mm_2026-11-19_noDipole`, and
two unrelated z-shift/ae1 variants exist in `plans/`). Building a new
plan from scratch requires the full offline pipeline (Layer 1 inverse
configuration → Layer 3 time-parameterized configuration → Jacobian
schedule build), which this session's own history shows can take
substantial offline compute time per candidate and does not always
converge on the first attempt (e.g. the S-curve @ 1.44T build in this
same broader session failed feasibility across 4 attempts). **Building
and feasibility-verifying two new scaled candidates was not completed
within this bounded pass's compute budget, and no numbers for them are
reported here** — fabricating screening results for plans that were not
actually built would violate this task's own instruction to select by
real sensitivity/feasibility evidence, not a desired outcome.

**Selected candidate: the existing 10×15mm rectangle**
(`plans/rectangle_10x15mm_skipglobal_L30to50mm_2026-11-19_noDipole`),
selected by a narrower but still evidence-based rule: it is the only
candidate with BOTH (a) already-confirmed Layer 1/Layer 3 feasibility
(it is the real plan this entire A–E campaign ran on) and (b) an
already-quantified, noise-exceeding Jacobian-sensitivity signal
(`STUDY2_PROPOSAL.md` §2: `(J_sched[k]-J0)@(z_pred[k]-z_ref[k])` RMS up
to 0.66mm in y, §2–4 already establish this exceeds the within-group
replay-computation spread). This selection rule is stated honestly as
weaker than the brief's intended three-way comparison — it reflects
feasibility by default (nothing else was built), not a demonstrated
superiority over 1.5×/2× candidates. Building and screening the scaled
candidates remains open future offline work, explicitly flagged, not
silently dropped.

## 3. Trial design

**Adaptation OFF** for both arms (`--jacobian-adaptation-mode off`),
matching conditions A/B/D, not E — isolates the scheduled-vs-frozen
question from online estimation.

**Scheduled-Jacobian MPC** (4 repetitions): identical to condition A's
settings.
```
python -m proper_research.hardware.online.rectangle_stage_a.run_mpc_delay_aware_insertion_anchor \
  --plan-dir plans/rectangle_10x15mm_skipglobal_L30to50mm_2026-11-19_noDipole/time_parameterized_configuration_path \
  --schedule-cache plans/rectangle_10x15mm_skipglobal_L30to50mm_2026-11-19_noDipole/jacobian_schedule_insertion_anchor.npy \
  --out-dir close_loop_logs --run-name jacobian_study_scheduled_repN_2026-XX-XX \
  --horizon 15 --jacobian-adaptation-mode off --input-penalty planned \
  --dipole-alignment-weight 0.0 --insertion-command-min-mm 20.0 --insertion-command-max-mm 70.0 \
  --measured-clearance-abort-mm 230.0 --beam-base-exclusion-floor-mm 230.0 \
  --magnet-exclusion-robust-margin-mm 3.0 --protective-config-reviewed
```

**Frozen-Jacobian MPC** (4 repetitions): identical, plus
`--freeze-jacobian --freeze-jacobian-index 0` (condition B's own
mechanism).

**Exact hashes** (both arms use the same two files):
```
840e5b4f44b98398374d63a6f93fbc20d006772a74f00729131e9a612c4cb5c2  proper_research/hardware/online/rectangle_stage_a/run_mpc_delay_aware_insertion_anchor.py
4346ef4e43e4620e59265459a05bfa830fce2123b4b77dcc54c93961cd8250b1  plans/rectangle_10x15mm_skipglobal_L30to50mm_2026-11-19_noDipole/jacobian_schedule_insertion_anchor.npy
```

## 4. Four-pair balanced execution order

Not the original campaign's fixed A-then-B block order (a stated
confound in the main `REPORT.md`'s §7). Concrete, specified order for the
8 total runs (alternating, scheduled first by arbitrary coin-flip
convention — record the actual flip if re-drawn before execution):

```
Repetition 1: scheduled, frozen
Repetition 2: frozen, scheduled
Repetition 3: scheduled, frozen
Repetition 4: frozen, scheduled
```

This balances which Jacobian condition occupies the earlier vs later
position within each repetition pair, across the 4 repetitions, so any
monotonic drift effect across a pair (rig warm-up, wear) is not
confounded with Jacobian choice the way the original campaign's fixed
order was.

## 5. Frozen statement

**This proposal is frozen pending review. No hardware has been run to
produce or validate it.** The algebraic correction in §1 is new,
verified evidence; the candidate screening in §2 is incomplete as
scoped, with the gap stated plainly rather than filled with invented
numbers; the trial design in §3–4 is ready to execute once the review
and the deferred 1.5×/2× screening (if still wanted) are resolved.
