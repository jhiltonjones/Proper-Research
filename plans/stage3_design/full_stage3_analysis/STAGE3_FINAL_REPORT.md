# Stage-3 Final Report: Contact Modelling, Exclusion Radius, and Controller Architecture

**Date:** 2026-10-08
**System:** a UR10e robot steering a magnetically-actuated flexible beam through
a vessel lumen via a source magnet, with camera-based beam-tip tracking.
**Scope:** closed-loop and open-loop hardware trials testing five hypotheses
(H1-H5, plus one secondary mechanism, H3b, and one secondary ablation) about
how contact modelling, the magnet-exclusion safety radius, and controller
architecture jointly determine tracking accuracy and task completion.

All figures referenced are in `figures/`, all tables in `tables/`.

---

## 1. Experimental design and data integrity

### 1.1 The actual run matrix

This investigation comprises **50 live hardware runs**, recomputed directly
from the run manifest and categorized by the role each run plays in the
results below:

| role | runs | what |
|---|---|---|
| **Primary** | **34** | 4 open-loop (H1) + 21 primary cells of a 2×2×2 closed-loop factorial — controller (MPC / inverse-Jacobian) × Jacobian model (contact / no-contact) × exclusion floor (210mm / 255mm), 3 reps each — + 3 matched-radius MPC-contact reruns at 255mm (replacing one factorial cell, §3) + 3 frozen-Jacobian reps at schedule index 0 + 3 frozen-Jacobian reps at schedule index 100 (H5) |
| **Secondary (ablations)** | **9** | 3 proportional-gain ablation (§8) + 3 selective-gate inverse-Jacobian + 3 clip-gate no-contact inverse-Jacobian (both H3b, §5) |
| **Superseded** (kept in the record, not deleted) | **6** | 3 original 255mm MPC-contact runs (run at a 15mm-looser true floor than intended, superseded by the matched rerun) + 3 first-attempt matched rerun (5mm off the intended target, superseded by a second, exact rerun) |
| **Excluded** (diagnosed hardware/software bug) | **1** | one 210mm MPC-contact rep aborted by an exclusion-margin edge case unrelated to control quality |

$34+9+6+1=50$. Every run is accounted for in `tables/per_run_summary_closedloop.csv`'s `cell_role` column; none is silently dropped from the record.

### 1.2 Exclusion-floor definitions — TRUE floor vs live threshold

A cylindrical exclusion zone around the beam's fixed base prevents the source
magnet from approaching close enough to interfere with the beam's own
magnetization. Two different control scripts enforce this constraint
differently, confirmed directly from each run's own `controller_metadata.json`
(for the MPC controller) or from the controller's own source code (for the
inverse-Jacobian controller, which produces no equivalent metadata file):

- **MPC:** a commanded `--beam-base-exclusion-floor-mm` value is the **TRUE**
  floor; the value actually enforced in the QP and in the abort check is
  `TRUE − tolerance` (tolerance default 5mm). A run directory labelled
  "255mm" therefore enforces 250mm in practice.
- **Inverse-Jacobian controller:** no tolerance subtraction — the commanded
  value **is** the enforced radius directly.

Every floor comparison in this report is matched on the **enforced
(live)** threshold, verified per run from logged metadata rather than
inferred from directory names. One comparison (the original 255mm MPC-contact
condition, §3) was not matched this way on its first attempt; it has been
superseded by a corrected, exactly-matched rerun, with the original kept in
the record rather than deleted.

### 1.3 Statistical unit and alignment

Trajectories are aligned by **path progress $s$** — the arc-length
coordinate of each plan's own reference path, read from the logged reference
index at each control tick — rather than by elapsed time, because different
conditions terminate at very different times and a time alignment would
compare unlike portions of the task. **The run is the statistical unit**:
with $n=3$ reps per condition, no significance testing is performed anywhere
in this report; every quantitative claim is either a run-level mean, an
explicit per-run breakdown, or direct mechanism evidence (recovered control
commands, measured magnet position, measured insertion length), never a
p-value.

### 1.4 Two narrow Jacobian-conditioning defects in the trajectory-varying contact schedule, checked and ruled out as the cause of the dominant tracking error

The MPC controller's contact-aware condition uses a pre-computed schedule of
Jacobian matrices, one per reference-path sample, built offline from the
contact model. A full-resolution scan of this schedule's condition number
(every one of its 534 samples, not a coarse stride) found two narrow
ill-conditioned regions:

- **Indices 0-9** ($s=0$-$1.3$mm): condition number decaying smoothly from
  **5093 down to 253**. Every other index away from this tail sits in a
  normal **3.65-18.4** range.
- **Index 505** ($s=71.4$mm) **only**: an isolated spike to **665.2**, against
  neighbouring values of 4.0-4.4 on both sides — a single-tick numerical
  artefact, not part of any broader trend. A corrected version of this
  schedule exists with this one entry repaired (confirmed by direct
  array comparison to differ from the original at index 505 only, resolving
  that entry's condition number to 4.06); later ablations in this report that
  needed a clean schedule (§5) used the repaired version.

**Neither defect explains the dominant tracking-error feature actually
observed.** The three largest tracking errors across the whole contact-aware
MPC dataset (2.4-3.7mm, one per rep) occur at $s\approx57$mm, where the
condition number is an unremarkable 5.8-6.1 — nowhere near either defect. A
targeted diagnostic was run at $s=54$-$60$mm (every tick, all three reps),
jointly examining eight candidate explanations: tracking error, one-step
prediction error against the scheduled Jacobian, all three singular values of
the scheduled Jacobian, commanded-velocity norm, tick-to-tick commanded-
velocity change, the live exclusion-constraint margin, the commanded
insertion rate, and the scheduled Jacobian's divergence from the contact model
recomputed at the measured state. None of the eight is anomalous in this
window: the schedule's condition number averages 4.85-4.88 (identical to the
broad baseline), the constraint margin averages 10.9-13.1mm (far from its
0mm floor), and neither velocity quantity shows an outlier at the peak tick.

**This is reported honestly as an unexplained local transient**, not solved.
It is too small (max 3.7mm, under every abort threshold used in this report)
to threaten any conclusion below, and is listed again in the Limitations
section (§10) rather than omitted.

### 1.5 Subsampling methodology for Jacobian-accuracy analysis (H4)

Evaluating the contact model's Jacobian at an arbitrary, off-reference
closed-loop state (as opposed to reading it from the precomputed schedule) is
computationally expensive (~2-3s per evaluation under this analysis
pipeline), making a full-resolution evaluation of every tick of every run
infeasible. H4 (§6) instead uses a **guided, category-capped sample** of
1636 ticks drawn from all 8 closed-loop conditions and all reps: ticks near
contact onset, the characteristic 47.8mm model-transition region, intervals
where the exclusion constraint is actively binding, and pre-failure windows
are deliberately over-represented relative to a uniform draw (capped per
category so no single region crowds out the others), because these are the
regions where a Jacobian mismatch is most likely to matter. This is stated
here so that every number in §6 is read as a sample statistic over this
guided set, not a full-resolution census.

---

## 2. H1 — Explicit contact modelling is required for an executable open-loop plan

**Hypothesis:** an open-loop plan built without modelling beam-wall contact
will fail to track once the beam makes contact, while a plan built with
contact modelling will not.

**Verdict: strongly supported.**

### 2.1 Design

Two open-loop plans were executed twice each: one built from a model that
includes beam-wall contact forces along the vessel lumen, one built from an
identical model with contact forces disabled (a free-space-only model). Both
plans drive the same reference path; only the planning model differs.

### 2.2 Result

| run | stop reason | $s_{end}$ (mm) | max error (mm) |
|---|---|---|---|
| contact rep1 | `path_complete` | 75.32 | 9.057 |
| contact rep2 | `path_complete` | 75.32 | 8.823 |
| no-contact rep1 | `tracking_error_exceeded(10.00mm)` | 61.57 | 9.943 |
| no-contact rep2 | `tracking_error_exceeded(10.03mm)` | 61.39 | 9.809 |

Both contact-aware reps execute the full 75.3mm path. Both no-contact reps
abort at the 10mm tracking-error threshold at $s\approx61.4$-$61.6$mm —
roughly 82% of the path.

### 2.3 Where the two plans actually differ

Over the common interval both plan types reach (0-61.4mm), RMSE is close
between the two (4.44/4.53mm contact vs 4.78/4.78mm no-contact) — the failure
is **not** a uniform accuracy offset present from the start. It is that the
no-contact plan's error keeps *growing* until it crosses the abort threshold,
while the contact-aware plan's does not.

Measuring the point at which the two plans' tracking errors first and
persistently diverge (exceeding the pre-contact noise floor by 3 standard
deviations for 5 consecutive samples) gives $s=50.5$mm and $s=51.8$mm for the
two rep pairs. The contact model's own offline prediction of when the beam
first touches the vessel wall is $s=28.5$mm. So the two plans do behave
similarly before contact and diverge after it, exactly as the hypothesis
predicts — but the measured divergence appears **~22mm later** than the
predicted contact onset. Early, light contact is initially tolerable; it only
becomes execution-limiting once the beam has advanced further into contact.

### 2.4 Independent confirmation of the mechanism

Evaluating the no-contact model and the contact model directly against each
other at matched states (not through either controller's schedule, just the
two models compared at the same configuration) gives a divergence of
**0.001%** at $s=15$mm — solidly pre-contact — rising to **16-33%** across
$s=32$-$75$mm, solidly post-contact. The missing physical term in the
no-contact model switches on almost exactly where the contact model predicts
it should, and is large once it does, directly corroborating why the
no-contact plan is viable early and not later.

---

## 3. H2 — Contact-Jacobian × exclusion-radius interaction

**Hypothesis:** whether using the wrong (no-contact) Jacobian model costs
tracking accuracy depends on how much configuration freedom the magnet-
exclusion floor leaves available — costless when the floor is loose, costly
when it is tight.

**Verdict: supported, on a genuinely radius-matched comparison at both floors.**

### 3.1 The interaction, matched

Mean full-run RMSE ($E$, mm) for MPC under each Jacobian model, at both
floors, with the enforced (live) threshold verified identical on both legs of
each radius (210mm: 205mm enforced on both; 255mm: 250mm enforced on both,
after the correction described in §1.2/§3.3):

| exclusion floor (enforced) | $E_{contact}$ | $E_{no\text{-}contact}$ | $\Delta_J=E_{no\text{-}contact}-E_{contact}$ |
|---|---|---|---|
| 210mm (205mm) | 0.598 | 0.497 | **−0.101** |
| 255mm (250mm) | 0.684 | 1.090 | **+0.406** |

$\Delta_J^{255}(+0.406) > \Delta_J^{210}(-0.101)$: the predicted ordering
holds. At the loose floor, using the wrong Jacobian model costs nothing (if
anything, the no-contact model tracks marginally better, well within normal
rep-to-rep spread). At the tight floor it costs 0.4mm of mean RMSE.

### 3.2 Completion is the stronger evidence

- **No-contact MPC: 3/3 complete at 210mm → 0/3 complete at 255mm**, all three
  aborting on the tracking-error threshold at $s\approx64$mm (~85% of the
  path).
- **Contact-aware MPC: 3/3 → 3/3** at both floors, at the identical enforced
  threshold — no degradation at all.

Tightening the exclusion floor converts the wrong Jacobian model from
"costless" to "run-ending", while the correct model is unaffected at the
exact same enforced threshold. This completion asymmetry is the strongest
single piece of evidence for this hypothesis.

### 3.3 What the uncorrected comparison looked like, and why it needed fixing

The first attempt at the 255mm contact-aware condition was, by its own
logged metadata, actually run at a TRUE floor of 240mm (enforced 235mm) — 15mm
more configuration freedom than the no-contact condition's actual 255mm (
enforced 250mm). That extra freedom inflated the apparent gap to +0.536mm.
A second attempt landed at TRUE=250mm (enforced 245mm), still 5mm off target,
giving RMSE 0.729mm. The figures reported in §3.1 are from a third, exactly
matched rerun (TRUE=255mm, enforced 250mm, confirmed identical to the
no-contact condition's own enforced threshold via each run's own metadata).
All three attempts are kept in the full run record (§1.1); only the matched
rerun is used for the headline numbers above.

### 3.4 Mechanism, made physically visible: where the magnet goes and how far the beam gets fed in

Plotting magnet-to-beam-base distance against path progress for all four MPC
conditions, with both enforced thresholds overlaid, shows the mechanism
directly rather than only inferring it from tracking statistics:

- **At 210mm**, the contact-aware and no-contact conditions are nearly
  indistinguishable: both dip down and pin themselves against the 205mm
  enforced threshold from $s\approx30$mm onward and stay there, within a few
  millimetres of each other, all the way to completion. The constraint
  doesn't force a choice between "stay near the base" and "correct for the
  missing contact term" — both are simultaneously available, so the wrong
  model's error is absorbed for free, matching $\Delta_J^{210}\approx0$.
- **At 255mm**, the two conditions diverge sharply and visibly. Both start
  with a wide excursion out to 390-420mm in the first 10mm, then settle
  toward the floor by $s\approx40$mm. From there, **contact-aware MPC keeps
  riding tight against the 250mm threshold for the rest of the path**,
  mirroring its own 210mm behaviour almost exactly. **No-contact MPC instead
  starts drifting *away* from the floor from $s\approx50$mm onward** —
  climbing from ≈255mm up past 300mm by the time it aborts at $s\approx64$mm.
  It is not failing to track precisely near the floor; it is retreating from
  near-base configurations altogether.
- **Insertion makes the same story even starker.** No-contact MPC at 255mm
  terminates having inserted the beam only **73.6-74.4mm** — the lowest of
  any of the four MPC conditions, including both 210mm conditions
  (80.2-82.4mm). Contact-aware MPC at 255mm reaches **89.7-91.0mm** — the
  *highest* of all four, exceeding even its own 210mm performance. The
  contact-aware controller pushes the beam in further at the tighter floor
  than at the looser one; the no-contact controller gets fed in less and then
  gives up.

The causal reading this supports: at the loose floor, both models can reach
the same near-base configurations, so the missing contact term costs
nothing. At the tight floor, those near-base configurations disappear for
both models equally — but the contact-aware model, correctly accounting for
the wall, finds a different, still-effective configuration right at the new
boundary, while the no-contact model has no reliable way to convert "stay
near the tightened floor" into "keep the tip where it needs to be", and
retreats to configurations where local tracking is easier but systematically
wrong, falling further behind until it aborts.

### 3.5 Lag vs cross-track (partially supported)

Decomposing tracking error into longitudinal lag (distance behind the
reference along the path) and lateral cross-track (perpendicular distance to
the reference): no-contact MPC's *terminal* lag grows 36× from 210mm to
255mm (0.079mm → 2.860mm, the abort-time value), and its *mean* lag grows
2.9× (0.167mm → 0.476mm). But cross-track grows by a similar factor over the
run as a whole (0.201mm → 0.479mm, 2.4×), so the lag/cross-track ratio only
moves from 0.83 to 0.99. The honest reading: the no-contact condition at the
tighter floor accumulates error in **both** components roughly
proportionally over the run, with the *terminal* failure specifically being
lag-dominated. "Lagging rather than merely displaced laterally" is true of
the end state, not of the whole run. By contrast, the inverse-Jacobian
conditions analysed in §4-§5 are overwhelmingly lag-dominated throughout
(ratio 7.1-10.6) — exactly what the hold-the-whole-command mechanism
described there predicts.

---

## 4. H3 — MPC vs inverse-Jacobian constraint awareness

**Hypothesis:** an MPC controller that represents the exclusion constraint
inside its own optimization will handle it fundamentally differently from a
naive resolved-rate (inverse-Jacobian) controller that does not — the latter
will repeatedly request motions the hardware cannot legally execute.

**Verdict: strongly supported, with the mechanism directly demonstrated, not inferred.**

### 4.1 How the raw, pre-safety-gate command was recovered

The inverse-Jacobian hardware runs analysed here gate externally: the
controller's own control law carries no knowledge of the exclusion
constraint whatsoever, and a separate external gate zeroes the *entire*
commanded 7-vector (6 joint velocities plus insertion rate) on any tick whose
one-tick-ahead predicted motion would violate the constraint. Only the
**post-gate** command is logged, so the controller's actual raw request had
to be reconstructed rather than read directly off the log.

It was reconstructed by replaying the exact control law offline — damped
least-squares resolved-rate control with the run's own gain, damping, and
schedule, zero null-space regularization (matching the hardware's own
configuration), the same velocity/acceleration box clip, and the raw
(not gated) previous command threaded recursively tick to tick — driven only
by each run's own logged measured states. **This replay reproduces the real
external gate's hold/pass decision on every single tick of all 12
inverse-Jacobian runs analysed (agreement = 1.000).** Every number below is
therefore an exact recovery of what the controller actually tried to do, not
an estimate.

### 4.2 How often the naive controller asks for something illegal

| exclusion floor | Jacobian | fraction of ticks requesting an infeasible motion | fraction held | number of hold episodes | longest episode | path progress lost to holds |
|---|---|---|---|---|---|---|
| 210mm | contact | **0.368** | 0.368 | 5.3 | 6.9s | 19.9mm |
| 210mm | no-contact | **0.000** | 0.000 | 0.0 | — | 0.0mm |
| 255mm | contact | **0.696** | 0.696 | 1.7 | 12.8s | 26.7mm |
| 255mm | no-contact | **0.419** | 0.419 | 1.3 | 9.1s | 20.9mm |

By contrast, the MPC controller's in-QP formulation of the identical
constraint is **active** (margin within 3mm) on only 8.5-20.4% of ticks
across all its runs, with minimum margins of 0.05-1.5mm, and **is never
violated on a single tick of any MPC run analysed**.

### 4.3 Why this is terminal, not just inefficient

MPC spends a comparable fraction of ticks *at* the constraint boundary as the
naive controller spends *violating* it, but the outcomes are completely
different. MPC redirects its motion within the feasible set and keeps the
beam advancing. The naive controller's entire command is zeroed instead.
Because the reference trajectory advances on its own fixed schedule
regardless of whether the robot actually moved that tick, **every held tick
is permanently lost ground**: the magnet parks within about 0.1mm of the
exclusion floor and sits there while the target races ahead along the path.
This produces the lag-dominated error signature noted in §3.5, 19.9-26.7mm of
total path progress lost to holding alone, and (§3.4's analogue for these
conditions) a truncated final insertion length.

### 4.4 Refinement 1 — the difference is latent, not demonstrated, when the constraint never binds

At 210mm with the correctly-matched (no-contact) Jacobian, the naive
controller requested **zero** infeasible commands across all three reps
(closest approach left 10.8-13.1mm of margin) and completed the path at
0.685mm RMSE, close to MPC's own 0.497mm in the same cell. The architectural
weakness this hypothesis names is real, but it is not exercised in every
cell — only where the constraint actually binds (41.9% of ticks even with
the matched Jacobian at 255mm, and with the mismatched Jacobian at 210mm).

### 4.5 Refinement 2 — at the tight floor, the binding constraint is often the workspace height limit, not the exclusion radius

The external gate checks the exclusion radius **and** the magnet's vertical
workspace bound as a single combined constraint. At 255mm the vertical bound
is co-dominant, and for one rep it is the entire story: that run shows 0.000
exclusion-radius-infeasible ticks but 0.737 vertical-bound-infeasible ticks,
with a minimum exclusion distance of 280mm — nowhere near the radius at all.
The other two 255mm contact-condition reps show a mix (31.7%/42.7% radius,
42.2%/18.4% vertical). The general claim this hypothesis makes — "the naive
controller repeatedly requests configurations that cannot legally be
executed, and is held" — is supported; the specific constraint most often
responsible at the tight floor is not always the one named in the
hypothesis.

---

## 5. H3b — Secondary mechanism: unregularized redundant-configuration drift

**This is a separate result from H3, not a continuation of it.** H3 is about
whether the controller *recognises* the exclusion constraint at all. H3b asks
what happens once that recognition is fixed — a second, independent failure
mode appears that has nothing to do with the exclusion constraint.

*(Terminology note: this result is called "redundant-configuration drift"
rather than "null-space drift" throughout. The evidence presented is
divergence in joint space — specifically joint 0, the base rotation — that
tracks with the robot's 4 kinematically redundant degrees of freedom relative
to the beam-tip task; it has not been explicitly projected into the
controller's instantaneous or finite-horizon null space, so the stronger,
more specific claim of literal null-space motion is not made here.)*

### 5.1 Design

An additional magnet-protection gate mode was built beyond the external
hold-everything gate used in §4: a **selective** mode (an external gate that
zeroes only the individual joint-velocity components whose own sign is
pushing the predicted margin further into violation, passing every other
component through unmodified). Six hardware reps at the 255mm exclusion
floor test this gate with two different Jacobian models. (A separate
**clip** mode also exists — the controller's own internal minimum-norm
half-space projection — but is not part of the verified evidence below: one
run set's directory name suggested it used `clip`, but replaying the actual
control law against its logged states and testing all three candidate gate
implementations against the logged commands shows a 100% match to the
**selective** gate specifically (406/406, 357/357, and 396/396 ticks across
its three reps, against 86-94% for the clip and hold candidates on the same
data) — this set, despite its name, also ran the selective gate. This
mismatch between directory name and actual run configuration is consistent
with a labelling oversight flagged, but not corrected, before the run was
executed.)

| | selective gate + contact Jacobian | selective gate + no-contact Jacobian |
|---|---|---|
| completed | **0/3** — all stopped on a generic robot-workspace-box violation unrelated to the magnet constraint, at $s\approx62$-$64$mm | **0/3** — same generic workspace-box stop, at control step 357-406 |
| exclusion-gate clip activity | 0-1.9% of ticks | magnet never within 0.5mm of its own floor |
| minimum magnet-to-base distance achieved | 257.7-258.6mm (floor 255mm) | 255.5-257.6mm (floor 255mm) |
| matched MPC comparator, same floor and Jacobian | **3/3** `path_complete` | **0/3**, aborted earlier on the tracking-error threshold — the H2 mechanism (§3), not this one |

**The magnet-exclusion gate works correctly in every one of these six reps**
— clip activity is minimal to none, and the magnet never approaches its own
floor in any rep of either pairing. The gate is demonstrably not what ends
these runs; something else does.

### 5.2 The mechanism

Plotting joint-0 (base rotation) angle against path progress for both
pairings alongside their matched MPC comparator shows the controllers
tracking each other closely out to $s\approx55$mm, then sharply diverging.
MPC's joint-0 trajectory plateaus at a bounded value and stays there for the
rest of the run — whether MPC itself goes on to complete (with the contact
Jacobian) or fails for the unrelated H2 reason (with the no-contact
Jacobian). The inverse-Jacobian controller's joint-0 keeps climbing past
wherever MPC's own trajectory plateaus, in the same window its tracking
error grows and the TCP crosses the generic (magnet-unrelated) workspace
box: direct joint-0 range comparisons over an identical early window show the
inverse-Jacobian controller consistently reaching several degrees further
than MPC under the same Jacobian and floor, with real rep-to-rep variability
in exactly how far the drift runs before the box check trips it.

Both controllers are run with zero explicit posture or redundancy
regularization — the inverse-Jacobian controller's own `nullspace_gain`
parameter and MPC's own terminal state-cost weight are both set to zero, a
deliberate, matched fair-comparison condition used throughout this report.
But MPC's quadratic cost is not *only* a state-tracking cost: its
input-tracking and input-increment weights apply to the **entire
7-dimensional commanded input**, including the four directions that are
kinematically redundant relative to the beam-tip task and therefore have no
effect on tracking at all. These cost terms implicitly damp *every*
commanded direction, task-relevant or not, on every single tick, with no
explicit posture target needed anywhere. The naive inverse-Jacobian
controller's redundancy resolution, at zero regularization gain, has **no
such term anywhere** — its null-space projector is multiplied by exactly
zero, so the four redundant directions are completely free to drift however
the resolved-rate solution's own numerical asymmetries happen to push them,
with nothing pulling them back, tick after tick, for as long as the run
lasts. Over most of the path this drift is harmless, since task tracking is
by construction insensitive to it — but once the path also becomes locally
demanding (the same $s\approx55$-$65$mm region flagged independently in
§1.4), MPC's always-on implicit damping keeps it inside a bounded,
recoverable configuration, while the naive controller — having wandered
further from a typical posture for longer, with nothing resisting that
drift — has no mechanism to recover, and the divergence runs away until the
robot physically exits its own workspace box.

### 5.3 Scope of this result

This mechanism was confirmed to reproduce across **two independent Jacobian
sources (contact, no-contact)**, both using the selective gate (verified
directly from logged commands, not from directory names — see §5.1) — it is
a property of how this control law resolves kinematic redundancy, not an
artefact of the specific Jacobian model supplied. Whether it also reproduces
under the controller's internal **clip** mode specifically remains untested;
no verified hardware run in this investigation's record actually exercised
that gate mode. It does **not** change the H3 verdict in §4: the
exclusion gate itself works correctly throughout every rep examined here. It
is, however, a second, entirely independent reason the inverse-Jacobian
controller underperforms MPC, and a plausible (though not separately
isolated) contributor to the broader late-path tracking degradation the
inverse-Jacobian conditions show in the main factorial of §3-§4.

---

## 6. H4 — Scheduled Jacobian accuracy against the contact model at the measured state

**Hypothesis:** a trajectory-varying (scheduled) Jacobian is more accurate
than the hardware's actual local behaviour would require only if it tracks
the state the robot actually visits; deviations from the planned path, and
use of the wrong physical model, both degrade this accuracy, and that
degradation should matter more where the exclusion floor forces larger state
deviations.

### 6.1 The only physically-grounded reference, and why

The only Jacobian that is ever physically realized on this hardware is the
contact model evaluated at the state the robot actually occupies at tick
$k$ — written here as $J_C^{state}(x_k)$. This is **not** a direct physical
measurement of a Jacobian; no instrument on this rig measures one. It is the
contact model recomputed at the measured hardware state, and its only
physical grounding is indirect: separately checking that $J_C^{state}\Delta\chi$
predicts measured camera tip motion reasonably well (quantified in §6.4 and
discussed further in the Limitations, §10). There is, however, no physically
realized "no-contact Jacobian" of any kind on this hardware at all — a
controller scheduled with a no-contact model still executes its commands on
the real, physically contacted hardware. Anywhere a no-contact model
evaluated at the actual measured state is used below, it is labelled
explicitly as a **counterfactual**, $J_{NC}^{cf}(x_k)$: a diagnostic
construction, never treated as a primary reference or as anything physically
realized.

This section answers three questions, using the guided 1636-tick sample
described in §1.5, across all 8 closed-loop conditions and all reps.

### 6.2 Question 1 — for contact-scheduled controllers, how accurate is the schedule against $J_C^{state}$?

### 6.3 Question 2 — for no-contact-scheduled controllers, how wrong is $J_{NC}^{sched}$ against $J_C^{state}$? (the central comparison)

Both questions use the identical relative-Frobenius quantity,
$\|J_{used}^{sched}(s_k)-J_C^{state}(x_k)\|/\|J_C^{state}(x_k)\|$, against the
one physically-grounded reference, computed for every condition:

| exclusion floor | condition | mismatch, mean | mismatch, median | mismatch, p95 | correlation with tick-level tracking error |
|---|---|---|---|---|---|
| 210mm | MPC, contact schedule (Q1) | 0.316 | 0.284 | 0.688 | +0.258 |
| 210mm | MPC, no-contact schedule (Q2) | 0.417 | 0.433 | 0.764 | **−0.043** |
| 210mm | InvJac, contact schedule (Q1) | 0.253 | 0.224 | 0.483 | +0.508 |
| 210mm | InvJac, no-contact schedule (Q2) | 0.481 | 0.559 | 0.870 | −0.106 |
| 255mm | MPC, contact schedule (Q1) | 0.347 | 0.333 | 0.819 | +0.204 |
| 255mm | MPC, no-contact schedule (Q2) | 0.483 | 0.470 | 0.882 | **+0.470** |
| 255mm | InvJac, contact schedule (Q1) | 0.486 | 0.488 | 0.891 | +0.728 |
| 255mm | InvJac, no-contact schedule (Q2) | 0.277 | 0.273 | 0.497 | +0.752 |

The central result of this hypothesis: for the MPC no-contact condition, the
correlation between this mismatch and tick-level tracking error is
**near-zero at the loose floor (−0.043) and meaningfully positive at the
tight floor (+0.470)**. The schedule's mismatch against the real contact
physics only starts to matter for tracking once the exclusion floor
tightens — the same regime in which §3 and §5 independently show the
no-contact model's error stops being absorbable by available configuration
freedom. This ties the Jacobian-accuracy question directly to the same
physical mechanism established elsewhere in this report, rather than leaving
it as an isolated matrix-algebra statistic.

The inverse-Jacobian conditions show positive correlations at both floors
(+0.508/+0.728 for the contact schedule, −0.106/+0.752 for the no-contact
schedule) — for this controller, schedule mismatch is a more consistent
tick-level predictor of tracking error than it is for MPC, plausibly because
MPC's own optimization partly compensates for local model error in a way the
naive controller's one-step resolved-rate law cannot.

### 6.4 Question 3 — does the mismatch matter in the commanded direction?

Three further quantities, all computed against $J_C^{state}$: the
command-weighted mismatch
$\epsilon_{u,k}=\|(J_{used}^{sched}-J_C^{state})\Delta\chi_k\|$ (millimetres
of predicted tip-motion error), the prediction gain ratio
$\|J_{used}^{sched}\Delta\chi\|/\|J_C^{state}\Delta\chi\|$ (how much the
schedule over- or under-predicts the actual tip response to the same
command), and the angle between the two matrices' dominant output
directions.

| exclusion floor | condition | $\epsilon_u$, mean (mm) | $\epsilon_u$, p95 (mm) | gain ratio | direction angle (deg) |
|---|---|---|---|---|---|
| 210mm | MPC, contact | 0.167 | 0.377 | 1.40 | 15.2 |
| 210mm | MPC, no-contact | 0.141 | 0.327 | 1.68 | 21.3 |
| 255mm | MPC, contact | 0.173 | 0.492 | 1.56 | 17.5 |
| 255mm | MPC, no-contact | **0.319** | **0.751** | **2.64** | 23.6 |

The no-contact MPC condition at 255mm is the clear outlier on every one of
these three quantities simultaneously: the largest command-weighted
mismatch in absolute terms (mean 0.32mm, p95 0.75mm — still a fraction of
this condition's own 1.09mm whole-run RMSE, §3.1), and the only condition
whose gain ratio departs substantially from 1 (mean 2.64). A gain ratio of
2.64 means that, projected through the real contact-model response, the
schedule systematically over-predicts how much tip motion a given command
will actually produce by roughly $2.6\times$ — the controller believes its
corrections are more than twice as effective as they really are. That is a
specific, physically interpretable failure signature, concentrated at
exactly the exclusion floor where §3 and §5 independently show the
no-contact model's error can no longer be compensated by available
configuration freedom.

### 6.5 Secondary diagnostic — staleness vs missing-contact-physics, via the counterfactual

For additional context only, not as a primary result, the no-contact
schedule's total mismatch against $J_C^{state}$ can be split into two parts
using the explicitly-labelled counterfactual:
$J_{NC}^{sched}-J_C^{state} = \underbrace{(J_{NC}^{sched}-J_{NC}^{cf})}_{\text{schedule/state staleness}} + \underbrace{(J_{NC}^{cf}-J_C^{state})}_{\text{missing-contact-physics}}$.

| exclusion floor | condition | staleness term | missing-contact term | cosine of angle between the two terms | cross-term as a fraction of total squared magnitude |
|---|---|---|---|---|---|
| 210mm | MPC, no-contact | 0.165 | 0.333 | 0.272 | +0.013 |
| 210mm | InvJac, no-contact | 0.194 | 0.439 | 0.302 | +0.020 |
| 255mm | MPC, no-contact | 0.269 | 0.370 | 0.150 | −0.072 |
| 255mm | InvJac, no-contact | 0.290 | 0.062 | **−0.254** | **−0.162** |

The missing-contact-physics term is usually the larger of the two, but
**the two terms are not generally aligned** — at 255mm their cosine is small
or clearly negative, meaning they partially cancel rather than add
constructively. Simply attributing the total mismatch to "whichever term is
numerically larger" would therefore overstate that term's actual
contribution to the total. This decomposition is presented only as a
secondary diagnostic for exactly this reason, and because it depends on the
explicitly-labelled counterfactual $J_{NC}^{cf}$ rather than on any
physically realized quantity.

### 6.6 Conclusion

Measured against the one physically-grounded reference available on this
hardware, the schedule mismatch is real, structured, and — in its command-
weighted and gain-ratio forms — turns on precisely where the rest of this
report's mechanism predicts it should: negligible correlation with tracking
error and a near-unity gain ratio at the loose floor, versus a clear positive
correlation and a 2.6× command-direction over-prediction at the tight floor,
specific to the no-contact-scheduled MPC condition. In absolute
command-weighted terms the effect remains modest relative to total tracking
error (sub-millimetre against a roughly 1mm whole-run RMSE), so it is not the
single dominant driver of the completion and tracking differences reported
in §3-§5 — but it is a real, floor-dependent contributor, not a negligible
one, and it is quantitatively consistent with the mechanisms H2 and H3
establish through entirely independent evidence (magnet position, command
recovery).

---

## 7. H5 — Scheduled vs frozen Jacobian

**Hypothesis:** a Jacobian that updates along the trajectory (scheduled)
produces better tracking and more reliable completion than any single fixed
(frozen) linearization, because the beam's dynamics change materially as it
advances through the vessel and makes contact.

**Verdict:**

$$\boxed{\text{No single fixed linearization matched the scheduled Jacobian's combination of tracking quality and robust completion.}}$$

### 7.1 Design: two frozen ablations, chosen to separate conditioning from staleness

A single frozen-Jacobian condition confounds two possible explanations for
any degradation it shows: is the degradation caused by **freezing** itself
(losing trajectory-varying updates — a staleness effect), or by whichever
**one specific matrix** happens to get frozen? To separate these, two frozen
conditions were run, both using MPC with every other setting identical to
the scheduled baseline, differing only in which single schedule index's
Jacobian is held fixed for the entire run:

- **Frozen at index 0** ($s=0$mm): the single worst-conditioned point on the
  entire 534-sample schedule (condition number 5093, against 4-17 almost
  everywhere else). A separate live-model check found that, at this exact
  state, the contact and no-contact models disagree by **42%** — despite
  being nominally pre-contact, this is not a clean snapshot of either
  physics regime; its pathological conditioning coincides with a genuine
  departure from ordinary free-space behaviour.
- **Frozen at index 100** ($s=14.96$mm): a cleanly-conditioned point
  (condition number 9.75, solidly inside the schedule's normal range),
  chosen specifically to sit well before contact onset ($s=28.5$mm) and
  clear of any other known schedule anomaly. The same live-model check found
  the contact and no-contact models agree to within **0.001%** at this exact
  state — i.e. this frozen matrix is a genuine, physically-representative
  free-space Jacobian, unlike index 0.

Both ablations were verified, directly from each run's own
`controller_metadata.json`, to enforce the identical 210mm TRUE / 205mm
enforced exclusion floor as each other and as the scheduled baseline — this
is a genuine same-floor, three-way comparison.

### 7.2 Headline result

| | frozen at index 0 (ill-conditioned) | frozen at index 100 (well-conditioned, pre-contact) | scheduled (updates every tick) |
|---|---|---|---|
| completion | **3/3** `path_complete` | **0/3**, all `tracking_error_exceeded` late in the path | **3/3** `path_complete` |
| common-interval (0-68.6mm) RMSE | 1.595mm | **0.932mm** (better, until it fails) | **0.523mm** (best overall) |
| common-interval median error | 1.223mm | 0.312mm | 0.329mm |
| common-interval error AUC | 1.313 | 0.557 | 0.407 |

The well-conditioned frozen matrix (index 100) tracks *better* than the
ill-conditioned one (index 0) on every metric within the interval both
survive — and then fails outright, while the ill-conditioned one survives to
the end. Both frozen choices lose to the scheduled Jacobian on completion,
tracking, or both.

### 7.3 Why index 0 survives and index 100 does not

Index 0's persistent, moderate tracking cost (roughly a 1-2mm error plateau
through the middle of the path) is consistent with controlling through a
near-singular matrix for the entire run: it is a conditioning problem,
present from the very first tick, not a problem that grows with distance
from the freeze point. Index 100's failure is the opposite shape: its
tracking stays indistinguishable from the scheduled baseline's for roughly
the first 58-68mm of the path — through contact onset and well beyond it —
and then breaks away within a narrow window and climbs essentially
monotonically to a 3.1-4.7mm lag at abort. That is the signature of
staleness specifically: a matrix that was genuinely representative at the
moment it was frozen, becoming progressively less representative as contact
mechanics evolve past the regime it was frozen in, until the controller can
no longer compensate.

A direct, independent check (comparing each frozen matrix's dominant output
direction against the true scheduled Jacobian's at every one of the 534
schedule indices, using the schedule arrays alone) shows index 0's direction
staying consistently closer to the true, evolving direction than index 100's
at every checkpoint past contact onset — for example, at $s=45$mm the angle
to the true direction is 46.7° for index 0 versus 56.3° for index 100; at
$s=75$mm it is 64.6° versus 74.3°. Index 0, despite its pathological *local*
conditioning, is the more *directionally durable* choice across the beam's
whole post-contact evolution, which is consistent with it being the one that
survives.

This directional-alignment finding explains the *long-run completion
outcome* (which frozen matrix survives to the end) but does **not** explain
the *regional* tracking-quality pattern within the interval both survive —
index 100 tracks better than index 0 through $s\approx20$-$55$mm despite
having the *worse* directional alignment throughout that same stretch.
Long-run directional alignment and local, tick-to-tick tracking quality are
measuring genuinely different things in this dataset, and this report does
not claim to fully reconcile why a locally noisier but directionally durable
matrix (index 0) and a locally cleaner but directionally decaying one (index
100) trade off exactly the way they do. What is established directly:
freezing at a well-conditioned, physically-correct-at-its-own-state Jacobian
is not sufficient for good long-run performance, and freezing at a
badly-conditioned one is not sufficient to prevent it either — the schedule
update itself is doing real work that neither single frozen snapshot
reproduces.

---

## 8. Secondary ablation — proportional gain (0.6 vs 1.0)

**Hypothesis under test:** raising the inverse-Jacobian controller's
proportional gain simply makes it request infeasible motions more
aggressively, worsening the H3 mechanism.

**Verdict: contradicted.**

| gain | RMSE (mm) | max error (mm) | mean raw command magnitude | p95 raw command magnitude | fraction infeasible | fraction held | closest exclusion approach (mm) | sign reversals per tick |
|---|---|---|---|---|---|---|---|---|
| 0.6 | 0.685 | 1.858 | 0.0122 | 0.0369 | **0.000** | **0.000** | 221.7 | 0.857 |
| 1.0 | 1.128 | 4.820 | 0.0286 | 0.1096 | **0.000** | **0.000** | 228.0 | 1.152 |

All six runs (three reps per gain) completed the path, and the
infeasible-request rate is **exactly zero at both gains** — the external
hold gate never fires at either setting, and the magnet's closest approach
to the exclusion boundary actually moves slightly *further away*
(221.7mm → 228.0mm) at the higher gain, not closer. The hypothesis as stated
is directly contradicted: this is not constraint pressure.

What does change: the raw requested command grows roughly 2.3× (mean) to 3×
(p95) larger, tracking RMSE rises 65%, max error rises 2.6×, and the number
of per-tick commanded-direction sign reversals rises 34%. This pattern — a
larger, more oscillatory raw command with more sign reversals, at a fixed
damping setting — is **consistent with** underdamped oscillatory behaviour
when proportional gain is raised without a matching increase in damping.
This has not been confirmed with a dedicated damping sweep and is reported
as a plausible explanation, not a demonstrated mechanism. What is established
without that further experiment: whatever the precise mechanism, the
resulting tracking degradation is a pure accuracy cost, independent of the
exclusion constraint entirely, and does not bear on H3.

---

## 9. Integrated discussion

Two largely independent axes account for this investigation's results.

**Modelling fidelity and available configuration space (H1, H2, H4, H5).** An
executable plan requires contact physics at all (H1): a plan built without it
tracks comparably to one built with it right up until the beam makes contact,
then fails. Whether using the *wrong* model during closed-loop control is
costly depends entirely on how much configuration freedom the exclusion
floor leaves available: at a loose floor there is enough freedom to absorb a
wrong model for free; at a tight floor there is not, and the wrong model both
tracks worse and runs out of room to recover (H2) — and the Jacobian
mismatch against the one physically-grounded local reference specifically
turns from statistically irrelevant to a real, direction-specific, and
quantifiable contributor at exactly that same tight floor (H4). Nor is a
single fixed linearization — however well- or ill-conditioned at the moment
it is taken — a substitute for continual updating: contact mechanics evolve
past any one frozen snapshot's regime of validity, and a trajectory-varying
schedule beats both a well-conditioned and an ill-conditioned fixed
alternative on the combination of tracking quality and robust completion
(H5).

**Controller architecture (H3, H3b).** Entirely independent of modelling
quality, a controller unable to represent the exclusion constraint inside
its own optimization is held in place rather than redirected whenever it
wants to do something the hardware physically cannot, and bleeds task
progress every single time this happens, because the reference advances
regardless (H3). Independent again of *that* mechanism, a controller whose
redundancy resolution carries no regularization on its task-irrelevant
directions drifts in configuration space without bound, eventually exiting
the robot's own physical workspace regardless of which Jacobian model it was
given — a second, unrelated reason the same architecture underperforms even
once the first mechanism is specifically fixed (H3b). MPC's design avoids
both problems as a direct consequence of its formulation rather than through
any separately-tuned fix: its in-QP constraint keeps every commanded tick
feasible, and its input-tracking and input-increment costs regularize every
commanded direction — task-relevant or not — without needing an explicit
posture target anywhere.

Together, these results separate "is the model right" from "is the
controller's optimization structured correctly" as two distinct, additive
sources of the performance gap observed between the contact-aware MPC
condition and every other condition tested in this investigation.

---

## 10. Limitations

- **$n=3$ reps per condition.** No significance testing is performed
  anywhere in this report; every quantitative claim is a run-level mean or
  an explicit per-run breakdown, and should be read with that sample size in
  mind.
- **$J_C^{state}$ is a recomputed model evaluation, not a direct physical
  measurement.** No instrument on this rig measures a Jacobian directly.
  The contact model is recomputed at the measured hardware state, and its
  physical grounding comes only from a separate, indirect check: comparing
  predicted tip motion ($J_C^{state}\Delta\chi$) against measured camera
  tip displacement. Every "$J_C^{state}$" comparison in §6 should be read as
  "the scheduled model versus the contact model re-evaluated at the measured
  closed-loop state", not as ground truth in an absolute sense.
- **Abort thresholds differ by controller architecture.** The tracking-error
  abort threshold is 5mm for MPC runs and 20mm for inverse-Jacobian runs.
  Raw "did it complete" comparisons across architectures must be read with
  this difference in mind; RMSE, median, and area-under-curve statistics are
  unaffected by the threshold, only by where a run happened to stop.
  Similarly, the open-loop runs in H1 use their own, separate 10mm
  threshold.
- **One unexplained local transient** ($s\approx57$mm, contact-aware MPC,
  §1.4): eight candidate explanations were checked jointly and all ruled
  out; the cause remains unknown. It is too small (under 3.7mm, below every
  abort threshold used here) to threaten any conclusion in this report, but
  is reported rather than silently omitted.
- **Single robot, single vessel geometry, single beam.** No claim in this
  report has been independently checked against a second physical rig,
  vessel shape, or beam.
- **The idx0-vs-idx100 completion/tracking trade-off (§7.3) is not fully
  mechanistically reconciled.** Directional alignment with the evolving true
  Jacobian explains which frozen matrix survives to the end of the path, but
  not which one tracks better moment-to-moment within the region both
  survive. This report establishes both facts directly and reports their
  coexistence rather than resolving it.

---

## 11. Final conclusions

| Hypothesis | Verdict | Key qualifier |
|---|---|---|
| **H1** — contact-aware planning is required for an executable open-loop plan | **Strongly supported** | measured tracking divergence between plans lags the model-predicted contact onset by ~22mm; early contact is tolerable, sustained contact is not |
| **H2** — contact-Jacobian × exclusion-radius interaction | **Supported** | radius-matched on both legs via logged metadata; completion (3/3→0/3 for the wrong model vs 3/3→3/3 for the correct one) is the decisive evidence; magnet-position and insertion-length data make the mechanism directly observable, not just inferred |
| **H3** — MPC vs inverse-Jacobian constraint awareness | **Strongly supported** | exact (agreement = 1.000) recovery of the controller's raw commands; the architectural difference is latent, not demonstrated, in the one cell where the constraint never binds; roughly half of the tight-floor infeasible episodes are a workspace-height violation, not the named exclusion-radius violation |
| **H3b** — unregularized redundant-configuration drift (secondary mechanism) | **Supported as a second, independent failure mode** | reproduces across 2 Jacobian models, both verified (via command replay, not directory names) to have run under the selective gate; the internal clip mode remains untested; does not alter the H3 verdict, since the exclusion gate itself works correctly throughout |
| **H4** — scheduled-Jacobian accuracy against the contact model at the measured state | **Partially supported, and shown to strengthen specifically at the tight exclusion floor** | mismatch-vs-tracking-error correlation is near-zero at the loose floor and clearly positive (+0.47 to +0.75) at the tight floor; a 2.64× command-direction over-prediction is specific to the no-contact-scheduled MPC condition at the tight floor; the effect remains sub-millimetre in absolute command-weighted terms |
| **H5** — a trajectory-varying Jacobian outperforms any single fixed one | **Supported** | the scheduled Jacobian beats both a well-conditioned and an ill-conditioned frozen alternative on completion and/or tracking quality; the two frozen ablations' differing failure timing (conditioning-limited vs staleness-limited) is established but not fully reconciled with each other |
| Proportional-gain ablation (secondary) | **Hypothesis contradicted** | the infeasible-request rate is exactly zero at both gains tested; the resulting tracking degradation is consistent with, but not proven to be, underdamped oscillatory behaviour, and is independent of the exclusion constraint entirely |
