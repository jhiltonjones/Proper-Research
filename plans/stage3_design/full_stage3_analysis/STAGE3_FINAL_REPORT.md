# Stage-3 Final Report: Contact Modelling, Exclusion Radius, and Controller Architecture

**Date:** 2026-10-08
**Status:** This is the consolidated final report, written once the investigation's
open threads were resolved. It presents the final experimental design, verified
results, mechanisms, and conclusions — not the sequence of corrections that
produced them. `STAGE3_ANALYSIS.md` is retained unchanged as the full analysis
and audit trail (how confounds were found and fixed, superseded interpretations,
exploratory detail); this document does not replace it and is not a patch on it.

---

## 1. Experimental design and data integrity

### 1.1 The actual run matrix

**50 live hardware runs** make up this investigation (recomputed directly from
the run manifest, not reused from an earlier count — an earlier draft's header
figure of "34" predates several groups added later and should not be cited):

| role | runs | what |
|---|---|---|
| **Primary** | **34** | 4 open-loop (H1) + 21 primary cells of the 2×2×2 closed-loop factorial (MPC/InvJac × contact/no-contact × 210/255mm, 3 reps each) + 3 matched-radius MPC-$J_C$@255mm (replaces one superseded factorial cell, §3) + 3 frozen-Jacobian idx0 + 3 frozen-Jacobian idx100 (H5) |
| **Secondary (ablations)** | **9** | 3 gain ablation (§8) + 3 selective-gate InvJac + 3 clip-gate no-contact InvJac (both H3b, §5) |
| **Superseded** (kept, not deleted) | **6** | 3 original unmatched 255mm MPC-$J_C$ (`floor240_increased`, TRUE floor 15mm looser than its sibling) + 3 first-attempt matched rerun (TRUE=250mm, 5mm off-target) |
| **Excluded** (diagnosed bug) | **1** | 210mm MPC-$J_C$ rep1, aborted on a since-fixed exclusion-margin edge case |

$34+9+6+1=50$. Every run above is accounted for in `tables/per_run_summary_closedloop.csv`'s `cell_role` column; none is silently dropped.

### 1.2 Exclusion-floor definitions — TRUE floor vs live threshold

Two different scripts enforce the magnet-to-beam-base exclusion constraint
differently, confirmed directly from each run's own `controller_metadata.json`
(MPC) or script source (inverse-Jacobian controller, which has no equivalent
metadata file):

- **MPC** (`run_mpc_delay_aware_vessel.py`): `--beam-base-exclusion-floor-mm`
  sets the **TRUE** floor; the live QP/abort threshold is
  `TRUE − magnet-exclusion-tolerance-mm` (default 5mm). A directory labelled
  "255mm" enforces 250mm live.
- **Inverse-Jacobian controller** (`run_inverse_jacobian_online_vessel.py`):
  no tolerance subtraction — the CLI value **is** the live enforced radius.

Every floor comparison in this report is matched on the **live** threshold,
verified per-run from metadata, not inferred from directory names. Where an
original comparison was not matched this way (the original 255mm MPC-$J_C$
cell, §3), it has been superseded by a corrected rerun; the original is kept
in the record, clearly marked, not deleted.

### 1.3 Statistical unit and alignment

Trajectories are aligned by **path progress $s$** (each plan's own `path_s_m`,
indexed by logged `ref_index`), not by time, because conditions terminate at
very different times. **The run is the statistical unit**: with $n=3$ per
condition, no significance testing is performed; all claims are run-level
patterns plus direct mechanism evidence (recovered commands, magnet position,
insertion length), not p-values.

### 1.4 Two narrow schedule-conditioning defects, checked and found not to be the cause of the dominant tracking error

A full-resolution scan of the contact Jacobian schedule found two narrow
ill-conditioned regions: indices 0-9 (condition number decaying 5093→253) and
an isolated single-index spike at index 505 (665.2, against 4.0-4.4 neighbours).
Both were checked directly against the hardware logs. Neither explains the
dominant tracking-error feature in the dataset: the three largest tracking
errors in the whole MPC-$J_C$ run set (2.4-3.7mm) occur at $s\approx57$mm,
where the condition number is an unremarkable 5.8-6.1. A targeted diagnostic at
$s=54$-$60$mm checked eight candidate explanations together (tracking error,
one-step prediction error, all three singular values of the scheduled
Jacobian, command norm, tick-to-tick command change, constraint margin,
insertion rate, schedule-vs-recomputed-model divergence) — none is anomalous
in that window. **This is reported as an unexplained local transient**, not
solved; it is too small (max 3.7mm, under every abort threshold) to threaten
any conclusion below. The index-505 defect has a confirmed, available fix
(`..._repaired.npy`); later runs in this campaign that needed a clean schedule
(the H3b ablations, §5) used it.

---

## 2. H1 — Contact-aware offline configuration planning is required

**Verdict: strongly supported.**

| run | stop reason | $s_{end}$ (mm) | max error (mm) |
|---|---|---|---|
| contact rep1 | `path_complete` | 75.3 | 8.8-9.1 |
| contact rep2 | `path_complete` | 75.3 | 8.8-9.1 |
| no-contact rep1 | `tracking_error_exceeded(10mm)` | 61.6 | 9.8-9.9 |
| no-contact rep2 | `tracking_error_exceeded(10mm)` | ~62 | 9.8-9.9 |

The contact-aware plan completes the full path; the no-contact plan fails at
~82% progress with a comparable max error at the point it aborts. Measured
divergence between the two plans' tracking onsets ($s\approx50.5$-$51.8$mm)
lags the contact-model-predicted contact onset ($s=28.5$mm) by ~22mm — contact
is initially tolerable, not immediately disqualifying, but eventually is.

---

## 3. H2 — Contact Jacobian × available configuration space

**Verdict: supported, on a genuinely radius-matched comparison at both floors.**

### 3.1 The interaction, matched

Mean full-run RMSE ($E$, mm), live thresholds verified identical on both legs
of each radius via `controller_metadata.json` (210mm: live=205mm both
conditions; 255mm: live=250mm both conditions, after the correction in §1.1):

| radius (live) | $E_{MPC,C}$ | $E_{MPC,NC}$ | $\Delta_J=E_{MPC,NC}-E_{MPC,C}$ |
|---|---|---|---|
| 210mm (205mm) | 0.598 | 0.497 | **−0.101** |
| 255mm (250mm) | 0.684 | 1.090 | **+0.406** |

$\Delta_J^{255}(+0.406) > \Delta_J^{210}(-0.101)$: the predicted ordering holds.
At the loose floor the no-contact Jacobian costs nothing; at the tight floor
it costs 0.4mm of mean RMSE.

### 3.2 Completion is the stronger evidence

- **MPC-$J_{NC}$: 3/3 → 0/3** as the floor tightens, all three aborting on
  `tracking_error_exceeded` at $s\approx64$mm.
- **MPC-$J_C$: 3/3 → 3/3**, no degradation, at the identical live threshold.

### 3.3 Mechanism, made physically visible: magnet configuration and insertion

`figures/h2_magnet_distance_and_insertion_story.png` shows this directly, not
just inferred from error statistics:

- **At 210mm**, $J_C$ and $J_{NC}$ ride the same 205mm live threshold
  throughout and are nearly indistinguishable — the constraint doesn't force
  a choice between "near the base" and "correct for the missing contact
  term", so the no-contact model's error is absorbed for free.
- **At 255mm**, MPC-$J_C$ finds a new, still-effective configuration at the
  tightened floor (riding 250mm, insertion reaching **89.7-91.0mm**, the
  highest of any MPC condition). MPC-$J_{NC}$ instead **retreats** from the
  floor from $s\approx50$mm onward (climbing past 300mm by abort) and its
  insertion is truncated to **73.6-74.4mm**, the lowest of any MPC condition.
  Lacking the correct contact model, it has no reliable way to convert
  "stay near the tightened floor" into "keep the tip where it needs to be",
  so it gives up advancing the beam at all in its final ~15% of progress.

### 3.4 Lag vs cross-track (partial support)

MPC-$J_{NC}$'s terminal lag grows 36× from 210→255mm, but cross-track grows by
a similar factor (2.4×) over the whole run, so the end-state is lag-dominated
but the run as a whole accumulates error in both components. "Lagging rather
than merely displaced" is true of the terminal failure, not the entire run.

---

## 4. H3 — MPC vs inverse-Jacobian constraint awareness (main result)

**Verdict: strongly supported**, with the mechanism directly demonstrated, not inferred.

### 4.1 Exact command recovery

The inverse-Jacobian runs gate externally (`--magnet-protection hold`): the
controller's own math has no knowledge of the exclusion constraint, and an
external gate zeroes the whole command on any tick a one-tick-ahead
prediction would violate it. The log stores only the post-gate command, so
the raw, pre-gate request was recovered by replaying the exact control law
offline from logged measured states. **The replay reproduces the real gate's
hold/pass decision on every tick of all 12 runs (agreement = 1.000)** — the
numbers below are exact recoveries.

### 4.2 Infeasible requests and the hold mechanism

| radius | condition | frac. infeasible | frac. held | episodes | longest | progress lost |
|---|---|---|---|---|---|---|
| 210 | InvJac-$J_C$ | 0.368 | 0.368 | 5.3 | 6.9s | 19.9mm |
| 210 | InvJac-$J_{NC}$ | 0.000 | 0.000 | 0 | — | 0 |
| 255 | InvJac-$J_C$ | 0.696 | 0.696 | 1.7 | 12.8s | 26.7mm |
| 255 | InvJac-$J_{NC}$ | 0.419 | 0.419 | 1.3 | 9.1s | 20.9mm |

MPC rides the same constraint on 8.5-20.4% of ticks and **never violates it
on any tick of any run**. The naive controller requests genuinely infeasible
motion on 36.8-69.6% of ticks once the constraint binds, and is held in place
every single time — each held tick is permanently lost ground, since the
reference advances on its own schedule regardless. This produces the
lag-dominated error signature, 19.9-26.7mm of lost progress, and the
truncated insertion seen for the InvJac conditions in §6's analogue table.

### 4.3 Two refinements, reported honestly

- **Latent, not demonstrated, when the constraint never binds**:
  InvJac-$J_{NC}$@210mm requested zero infeasible commands (10.8-13.1mm of
  margin throughout) and completed. The architectural weakness exists but
  this cell doesn't exercise it.
- **At 255mm the binding constraint is often the z-workspace bound, not the
  exclusion radius** — one rep is a pure z-workspace lockup (0% radius
  violations, 73.7% z violations, never closer than 280mm to the exclusion
  radius). H3 as stated names the radius specifically; the general claim
  ("the naive controller repeatedly requests illegal configurations and is
  held") is supported, but not always via the named constraint.

---

## 5. H3b — Secondary: redundant configuration drift is a second, independent inverse-Jacobian failure mechanism

**This is a separate result from H3, not a continuation of it.** H3 is about
whether the controller *recognises* the exclusion constraint. H3b is about
what happens once that recognition is fixed — a second, unrelated failure mode
appears.

*(Terminology note: this report uses "redundant configuration drift" rather
than "null-space drift" — the evidence here is joint-space divergence,
specifically in joint 0, consistent with the controller's redundant degrees of
freedom being unregulated; it has not been explicitly projected into the
instantaneous or horizon null space, so the stronger claim ("null-space
motion") is not made.)*

### 5.1 The experiment

Two gate modes were built beyond `hold`: `clip` (the controller's own internal
minimum-norm projection) and `selective` (an external gate that clips only the
joint components whose sign worsens the margin, passing the rest through
unmodified). Six reps at the 255mm floor test two independent Jacobian/gate
pairings:

| | InvJac-selective + $J_C$ | InvJac-clip + $J_{NC}$ |
|---|---|---|
| completed | **0/3**, all `tcp_out_of_workspace`, $s\approx62$-$64$mm | **0/3**, all `tcp_out_of_workspace`, step 357-406 |
| gate (exclusion) clip activity | 0-1.9% of ticks | magnet never closer than 0.5mm to the floor |
| min magnet distance | 257.7-258.6mm (floor 255mm) | 255.5-257.6mm (floor 255mm) |
| matched MPC comparator | 3/3 `path_complete` | 0/3, `tracking_error_exceeded` (a different, earlier, H2-consistent failure) |

**The magnet-exclusion gate works correctly in every rep of both pairings** —
clip activity is minimal to none, and the magnet never approaches its own
floor. Something else ends these runs.

### 5.2 The mechanism

Overlaying joint-0 (base rotation) against path progress, both inverse-Jacobian
pairings track their MPC comparator closely out to $s\approx55$mm, then
diverge sharply: MPC's joint-0 plateaus and stays bounded (whether MPC itself
completes, as with $J_C$, or fails for the separate H2 reason, as with
$J_{NC}$); the inverse-Jacobian controller's keeps climbing past where MPC
ever goes, in the same window the TCP crosses the generic
(magnet-unrelated) workspace box. Both controllers run with zero explicit
posture regularization (`nullspace_gain=0` / $Q_N=0$) — but MPC's input cost
($R=700R_0$) applies to the full 7-dimensional command, including the 4
redundant directions, implicitly damping them every tick with no posture
target needed. The inverse-Jacobian controller's redundancy resolution has no
such term anywhere at `nullspace_gain=0` — nothing resists those directions
drifting, and when the path also becomes locally demanding ($s\approx55$-$65$mm,
the same region flagged in §1.4), the drift has nowhere to recover from.

### 5.3 Scope of this result

Reproduces across **2 Jacobian sources × 2 gate implementations** (contact/
selective and no-contact/clip), so it is a property of the control law's
redundancy handling, not an artefact of one Jacobian model or one gate. It
does **not** change the H3 verdict — the exclusion gate works throughout —
but it is a second, independent reason the inverse-Jacobian controller
underperforms MPC, and a plausible (not separately isolated) contributor to
its broader late-path degradation in the main factorial.

---

## 6. H4 — Scheduled Jacobian vs. the contact model recomputed at the measured state

**Framing.** The only Jacobian that is ever physically realized on this
hardware is the contact model evaluated at the state the robot actually
visited at tick $k$ — call it $J_C^{state}(x_k)$. This is **not** a direct
physical measurement of a Jacobian (no such instrument exists here); it is
the contact model recomputed at the measured hardware state, and its physical
grounding comes only indirectly, from checking $J\Delta\chi$ against measured
camera tip motion (§10). There is no physically realized "live no-contact
Jacobian" — a no-contact-schedule controller's commands are still executed on
the real, contacted hardware. Where a no-contact model evaluated at the
actual state is used below, it is labelled explicitly as a **counterfactual**,
$J_{NC}^{cf}(x_k)$, and used only as a secondary diagnostic — never as "live"
and never as a primary comparison target.

This section answers exactly three questions, using a guided, category-capped
sample of 1636 ticks across all 8 conditions and all reps (onset, the
47.8mm transition, constraint-active, and pre-failure regions oversampled
relative to a uniform draw; `tables/h4_final_mismatch_vs_Cstate.csv`).

### 6.1 Q1 — For contact-schedule controllers, how accurate is the schedule against $J_C^{state}$?

### 6.2 Q2 — For no-contact-schedule controllers, how wrong is $J_{NC}^{sched}$ against $J_C^{state}$? (the central comparison)

Both questions use the same relative-Frobenius quantity,
$\|J_{used}^{sched}(s_k)-J_C^{state}(x_k)\|/\|J_C^{state}(x_k)\|$, against the
one physically-grounded reference, for every condition:

| radius | condition | mismatch mean | median | p95 | corr. with tracking error |
|---|---|---|---|---|---|
| 210 | MPC-$J_C$ (Q1) | 0.316 | 0.284 | 0.688 | +0.258 |
| 210 | MPC-$J_{NC}$ (Q2) | 0.417 | 0.433 | 0.764 | **−0.043** |
| 210 | InvJac-$J_C$ (Q1) | 0.253 | 0.224 | 0.483 | +0.508 |
| 210 | InvJac-$J_{NC}$ (Q2) | 0.481 | 0.559 | 0.870 | −0.106 |
| 255 | MPC-$J_C$ (Q1) | 0.347 | 0.333 | 0.819 | +0.204 |
| 255 | MPC-$J_{NC}$ (Q2) | 0.483 | 0.470 | 0.882 | **+0.470** |
| 255 | InvJac-$J_C$ (Q1) | 0.486 | 0.488 | 0.891 | +0.728 |
| 255 | InvJac-$J_{NC}$ (Q2) | 0.277 | 0.273 | 0.497 | +0.752 |

The central result: for MPC-$J_{NC}$, the correlation between this mismatch
and tick-level tracking error is **near-zero at 210mm (−0.043) and
meaningfully positive at 255mm (+0.470)**. The schedule mismatch against the
real contact physics only starts to matter for tracking once the floor
tightens — exactly the regime where H2/H3 independently show the no-contact
model's error stops being absorbable (§3, §5). This is a materially cleaner
and more interpretable result than treating "mismatch vs a recomputed
no-contact model" as the target: it ties the Jacobian-accuracy question
directly to the same physical mechanism (loss of near-base configuration
freedom at the tight floor) established elsewhere in this report.

### 6.3 Q3 — Does the mismatch matter in the commanded direction?

Three command-relevant quantities, all computed against $J_C^{state}$:
$\epsilon_{u,k}=\|(J_{used}^{sched}-J_C^{state})\Delta\chi_k\|$ (mm),
the prediction gain ratio $\|J_{used}^{sched}\Delta\chi\|/\|J_C^{state}\Delta\chi\|$,
and the dominant-direction angle between the two matrices.

| radius | condition | $\epsilon_u$ mean (mm) | $\epsilon_u$ p95 (mm) | gain ratio | angle (deg) |
|---|---|---|---|---|---|
| 210 | MPC-$J_C$ | 0.167 | 0.377 | 1.40 | 15.2 |
| 210 | MPC-$J_{NC}$ | 0.141 | 0.327 | 1.68 | 21.3 |
| 255 | MPC-$J_C$ | 0.173 | 0.492 | 1.56 | 17.5 |
| 255 | MPC-$J_{NC}$ | **0.319** | **0.751** | **2.64** | 23.6 |

Figure: `figures/h4_final_mismatch_vs_Cstate_255mm.png` — tracking error,
mismatch, $\epsilon_u$, gain ratio, and direction angle vs path progress, for
MPC-$J_C$ vs MPC-$J_{NC}$ at 255mm. MPC-$J_{NC}$'s commanded-direction error
is both the largest in absolute terms (mean 0.32mm, p95 0.75mm — still a
fraction of the 1.09mm whole-run RMSE this condition posts, §3.1) and the
only one whose **gain ratio departs substantially from 1** (2.64 mean): the
no-contact-scheduled command, projected through the real contact-model
response, systematically over-predicts how much tip motion the commanded
input will produce. That is a specific, physically interpretable failure
signature — not just "the matrices disagree", but "the controller believes
its commands are more effective than they actually are", concentrated at
the floor where H2/H3 show the no-contact model can no longer be compensated.

### 6.4 Secondary diagnostic: staleness vs missing-contact-physics (counterfactual decomposition)

For context only — not a primary result — the no-contact schedule's mismatch
against $J_C^{state}$ decomposes as
$J_{NC}^{sched}-J_C^{state} = (J_{NC}^{sched}-J_{NC}^{cf}) + (J_{NC}^{cf}-J_C^{state})$,
schedule/state staleness vs the counterfactual missing-contact term:

| radius | condition | staleness term | missing-contact term | cos(angle) | cross-term frac. of total |
|---|---|---|---|---|---|
| 210 | MPC-$J_{NC}$ | 0.165 | 0.333 | 0.272 | +0.013 |
| 210 | InvJac-$J_{NC}$ | 0.194 | 0.439 | 0.302 | +0.020 |
| 255 | MPC-$J_{NC}$ | 0.269 | 0.370 | 0.150 | −0.072 |
| 255 | InvJac-$J_{NC}$ | 0.290 | 0.062 | **−0.254** | **−0.162** |

The missing-contact term is usually the larger of the two, but **the two
terms are not always aligned** — at 255mm the cosine between them is small
or negative, meaning they partially cancel rather than add; attributing the
total mismatch to "whichever term is larger" would overstate the smaller
term's actual contribution. This decomposition is kept only as a secondary
diagnostic, consistent with its dependence on the explicitly-labelled
counterfactual $J_{NC}^{cf}$ rather than any physically realized quantity.

### 6.5 Conclusion for H4

The schedule mismatch against the one physically-grounded reference
($J_C^{state}$) is real, structured, and — unlike earlier framings of this
question — **shown to turn on exactly where the rest of this report's
mechanism predicts it should**: negligible correlation with tracking error at
the loose floor, a clear positive correlation and a 2.6× command-direction
over-prediction at the tight floor, for the no-contact-scheduled controller
specifically. In absolute command-weighted terms the effect remains modest
relative to total tracking error (sub-mm against a ~1mm RMSE) — so it is not
the single dominant driver of H2/H3's results — but it is no longer
accurately described as "too small to matter"; it is a real, radius-dependent
contributor consistent with, and quantitatively supportive of, the
mechanism H2 and H3 establish independently.

---

## 7. H5 — Scheduled vs frozen Jacobian at 210mm

**Verdict:**

$$\boxed{\text{No single fixed linearization matched the scheduled Jacobian's combination of tracking quality and robust completion.}}$$

Both frozen ablations were verified, directly from each run's own
`controller_metadata.json`, to be at the identical TRUE=210mm/live=205mm
floor as each other and as the scheduled baseline — this is a genuine
same-floor, three-way comparison, not a confounded one.

| | frozen idx0 (ill-conditioned, s=0) | frozen idx100 (well-conditioned, pre-contact) | scheduled $J_C$ |
|---|---|---|---|
| completion | **3/3** `path_complete` | **0/3**, all `tracking_error_exceeded` late in the path | **3/3** `path_complete` |
| common-interval (0-68.6mm) RMSE | 1.595mm | **0.932mm** (better, until it fails) | **0.523mm** (best overall) |
| common-interval median | 1.223mm | 0.312mm | 0.329mm |

- **idx0** freezes at the one pathologically ill-conditioned point on the
  whole schedule (condition number 5093 vs 4-17 elsewhere) and at a point the
  live contact/no-contact models disagree by 42% — not a clean snapshot of
  any single physics regime. It survives the whole path, but pays a
  persistent, moderate tracking cost (1-2mm plateau) for most of it.
- **idx100** freezes at a clean, well-conditioned (condition number 9.75), and
  — independently confirmed via a live-model probe — genuinely
  free-space-representative pre-contact Jacobian. It tracks *better* than
  idx0 for most of the shared interval, right up until contact mechanics
  evolve far enough past that frozen snapshot's validity that it fails
  outright, late in the path.

Both frozen choices lose to the scheduled Jacobian on completion, tracking,
or both. **This report does not attempt to fully explain why idx0 survives
and idx100 doesn't** — the audit trail (`STAGE3_ANALYSIS.md` §7.7-7.8) explores
a directional-alignment-vs-conditioning-noise hypothesis in more depth, but
it is offered there as interpretation, not as an established mechanism, and
is not needed to support the boxed conclusion above.

---

## 8. Gain ablation (0.6 vs 1.0) — secondary

**Verdict: the specific hypothesis is contradicted.** Raising gain does not
increase infeasible-request rate (exactly 0.000 at both gains, inverse-Jacobian
$J_{NC}$@210mm) or push the magnet closer to its exclusion floor (221.7→228.0mm,
i.e. further away). It does cost tracking accuracy (RMSE +65%, max error
2.6×) and raises command magnitude (~2.3× mean) and per-tick sign-reversal
rate (+34%) — consistent with underdamped/oscillatory behaviour when gain
rises without a matching increase in damping, but this has not been confirmed
with a damping sweep and is not treated as warranted on its own. What is
established without qualification: this degradation is **independent of
constraint pressure** and does not speak to H3.

---

## 9. Integrated discussion

Two largely independent axes explain this campaign's results:

**Modelling fidelity and available configuration space (H1, H2, H4, H5).**
An executable plan needs contact physics (H1). Whether a wrong model is
*costly* depends on the exclusion floor: at a loose floor there is enough
configuration freedom to absorb the wrong model for free; at a tight floor
there isn't, and the no-contact model both tracks worse and runs out of
room to recover (H2), with the schedule-vs-actual-state Jacobian mismatch
specifically turning from irrelevant to a real, direction-specific
contributor at exactly that same tight floor (H4). No single fixed
linearization — however well- or ill-conditioned — matches a trajectory-
varying schedule's combination of tracking quality and robust completion,
because contact mechanics evolve past any single frozen snapshot's regime
of validity (H5).

**Controller architecture (H3, H3b).** Independent of modelling quality, a
controller that cannot represent the exclusion constraint in its own
optimization is held, not redirected, and bleeds progress every time it
tries something the hardware cannot do. Independent *again* of that,
a controller whose redundancy resolution has no regularizer on its
task-irrelevant directions drifts unboundedly in configuration space,
eventually exiting the robot's physical workspace regardless of which
Jacobian it was given — a second, unrelated reason the same controller
underperforms even when the first mechanism is fixed.

MPC avoids both: its in-QP constraint formulation keeps it feasible at every
tick (H3), and its input-tracking/increment cost regularizes every commanded
direction, task-relevant or not, without needing an explicit posture term
(H3b) — a property that falls out of the cost structure, not something
separately tuned for this purpose.

---

## 10. Limitations

- **$n=3$ per condition.** No significance testing is performed anywhere in
  this report; every quantitative claim is a run-level mean or an explicit
  per-run breakdown, and should be read as such.
- **$J_C^{state}$ is a recomputed model evaluation, not a direct physical
  measurement.** No instrument in this rig measures a Jacobian directly. The
  contact model is recomputed at the measured hardware state, and its
  physical grounding comes only from a separate, indirect check: comparing
  predicted $J\Delta\chi$ against measured camera tip displacement. Treat
  every "$J_C^{state}$" comparison in §6 as "scheduled model vs. contact
  model re-evaluated at the measured closed-loop state", not as ground truth
  in an absolute sense.
- **Abort thresholds differ by architecture** (`max_tracking_error_m` = 5mm
  for MPC runs, 20mm for inverse-Jacobian runs). Raw "did it complete"
  comparisons across architectures must be read with this in mind; RMSE,
  median, and AUC statistics are unaffected by the threshold, only by where
  each run stopped.
- **One unexplained local transient** (§1.4, $s\approx57$mm, MPC-$J_C$): eight
  candidate explanations were checked and ruled out; the cause remains
  unknown. It is too small to threaten any conclusion here, but is reported
  rather than silently omitted.
- **Single robot, single vessel geometry, single beam.** No claim in this
  report has been checked against a second physical setup.

---

## 11. Final conclusions

| Hypothesis | Verdict | Key qualifier |
|---|---|---|
| **H1** — contact-aware planning required | **Strongly supported** | divergence onset lags predicted contact onset by ~22mm |
| **H2** — $\Delta_J$ × exclusion-radius interaction | **Supported** | radius-matched on both legs; completion (3/3→0/3 vs 3/3→3/3) is the decisive evidence; magnet-position/insertion data makes the mechanism directly visible |
| **H3** — MPC vs InvJac constraint awareness | **Strongly supported** | exact (agreement=1.000) command recovery; latent when the constraint never binds; ~half of 255mm episodes are z-workspace, not radius |
| **H3b** — redundant configuration drift (secondary) | **Supported as a second, independent mechanism** | reproduces across 2 Jacobian sources × 2 gate modes; does not alter the H3 verdict |
| **H4** — scheduled-vs-actual-state Jacobian accuracy | **Partially supported, strengthened at the tight floor** | mismatch-vs-tracking-error correlation is null at 210mm, positive (+0.47-+0.75) at 255mm; command-direction over-prediction (gain ratio 2.64) specific to MPC-$J_{NC}$@255mm; still sub-mm in absolute command-weighted terms |
| **H5** — trajectory-varying Jacobian beats any fixed one | **Supported** | scheduled beats both frozen ablations on completion and/or tracking; no attempt made to fully explain idx0-vs-idx100's differing failure timing beyond conditioning vs staleness |
| Gain ablation (secondary) | **Contradicted as stated** | infeasible rate is exactly 0 at both gains; degradation consistent with, not proven to be, underdamped/oscillatory behaviour; independent of constraint pressure |
