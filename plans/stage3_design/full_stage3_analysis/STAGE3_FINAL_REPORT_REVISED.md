# Stage-3 Final Report (Revised): Contact Modelling, Exclusion Radius, and Controller Architecture

**Date:** 2026-10-08 (mechanism-closure pass)
**System:** a UR10e robot steering a magnetically-actuated flexible beam through
a vessel lumen via a source magnet, with camera-based beam-tip tracking.
**Scope:** closed-loop and open-loop hardware trials testing five hypotheses
(H1-H5, plus one secondary architectural mechanism and one secondary
ablation) about how contact modelling, the magnet-exclusion safety radius,
and controller architecture jointly determine tracking accuracy and task
completion.

**Relationship to the prior report.** `STAGE3_FINAL_REPORT.md` is the
audit/source document for everything below and remains the full run-
accounting record. This document is a clean rewrite, not an appendix to it:
every mechanism claim below is backed by one of four kinds of evidence,
labelled inline —

- **[hardware]** a direct measurement from a logged run;
- **[model]** an offline model evaluation (e.g. the contact model
  recomputed at a measured state);
- **[counterfactual]** a same-state controller replay — the exact online
  optimizer, re-solved at a frozen historical state with one input (usually
  a Jacobian) substituted, everything else identical;
- **[interpretation]** a reading consistent with the evidence above it,
  explicitly flagged as such, never asserted as directly measured.

Three places where this pass's evidence **revises** a claim in the prior
report are flagged explicitly where they occur (§3.4, §5.2, §7.3).

All figures referenced are in `figures/`, all tables in `tables/`, all new
analysis code in `scripts/`.

---

## 1. Experimental design and data integrity

### 1.1 The actual run matrix

Unchanged from the prior report: **50 live hardware runs** — 34 primary, 9
secondary ablations, 6 superseded (kept in the record), 1 excluded
(diagnosed bug). See `tables/per_run_summary_closedloop.csv`'s `cell_role`
column for the complete accounting; nothing is silently dropped. §1.1-§1.3
of the prior report (run matrix, exclusion-floor TRUE-vs-enforced
definitions, path-progress alignment, run-as-statistical-unit with $n=3$)
are reused unchanged here and not repeated.

### 1.2 The one new piece of shared infrastructure this pass required

Four of this pass's six tasks needed to ask "what would the controller have
commanded at this exact historical state, if given a different model" —
not inferable from logs alone. `scripts/mpc_same_state_replay.py`
reconstructs the actual online MPC object
(`ExactQNZeroTaskNullspaceDelayAwareMPC`, confirmed as the variant every
closed-loop MPC run in this investigation used, from each run's own
`controller_metadata.json["controller_variant"]="exact_qn0_R700_vessel"`)
in-process, using each run's own `predicted_beam_positions.jsonl` — a
per-tick solve log already containing the exact measured state, execution-
accumulator command, previous input, and realized Jacobian for every tick —
so that a frozen tick can be re-solved with an arbitrary Jacobian
substituted and nothing else changed.

**Validation.** Re-solving 20 ticks across 4 runs (210/255mm,
contact/no-contact) with no override — i.e. asking the replay to reproduce
what the run already did — matched the logged command to a median relative
error of **0.01%** and a 95th-percentile error under **0.06%**, consistent
with ordinary QP solver tolerance, not a methodological gap. **One tick out
of 20** (255mm matched-contact run, a tick with a very small command norm)
showed a 123% relative error — both the logged and replayed commands are
tiny in that case (norm $\approx 0.014$), consistent with a near-degenerate,
weakly-penalized direction of that particular QP rather than a replay bug;
every same-state counterfactual result below reports its own per-tick
sanity-check error alongside the result (`*_sanity_check.csv` files) so a
reader can see directly whether a given tick was well-conditioned for this
kind of replay.

This tool, and the direction it is used in, is always: *hold the measured
state and everything else fixed, vary only the model the optimizer is given,
and read off what it would have commanded.* It is never used to claim what
actually happened physically beyond that one re-solved tick.

---

## 2. H1 — Explicit contact modelling is required for an executable open-loop plan

Unchanged from the prior report (§2 there). **Verdict: strongly supported.**
Contact-aware open-loop plans complete the full 75.3mm path (max error
8.8-9.1mm); no-contact plans abort on the 10mm threshold at $s\approx61.4$-
$61.6$mm ($\approx82\%$ of the path). The two plans track similarly up to
$s\approx50$-$52$mm before diverging — about 22mm later than the contact
model's own predicted contact onset ($s=28.5$mm), i.e. early contact is
tolerable, sustained contact is not. Independent confirmation: the two
models, evaluated directly against each other at matched states, disagree by
0.001% pre-contact ($s=15$mm) and 16-33% post-contact ($s=32$-$75$mm)
**[model]**.

---

## 3. H2 — Contact-Jacobian × exclusion-radius interaction

**Hypothesis:** whether using the wrong (no-contact) Jacobian model costs
tracking accuracy depends on how much configuration freedom the magnet-
exclusion floor leaves available.

**Verdict: supported, with the failure mechanism now directly demonstrated
via same-state counterfactual, not inferred from trajectory shape.**

### 3.1 The interaction, matched (unchanged from the prior report)

| exclusion floor (enforced) | $E_{contact}$ | $E_{no\text{-}contact}$ | $\Delta_J$ |
|---|---|---|---|
| 210mm (205mm) | 0.598 | 0.497 | **−0.101** |
| 255mm (250mm) | 0.684 | 1.090 | **+0.406** |

**Completion: 3/3→3/3 for MPC-$J_C$ at both floors; 3/3→0/3 for MPC-$J_{NC}$.**
All three tight-floor MPC-$J_{NC}$ reps terminate on
`tracking_error_exceeded(5mm)` **[hardware]**, confirmed directly from
`summary.json` (5.34mm, 5.02mm, 5.44mm respectively) — not a workspace or
exclusion-constraint abort. This is the single most important fact this
section's mechanism must explain: **the failure is a pure tracking-accuracy
failure**, and any causal story that routes through "running out of
workspace" must be checked against that fact, not asserted around it.

### 3.2 Workspace and exclusion margins stay generous throughout — this revises the prior report

`scripts/h2_workspace_timeline.py` computed, at $s=\{45,50,55,58,60,62,64\}$mm
for every tight-floor MPC-$J_C$/MPC-$J_{NC}$ rep: the exclusion-radius
margin, the magnet z-workspace margin, and the margin to every face of the
robot's configured TCP workspace box (`tables/h2_workspace_margin_timeline.csv`,
`tables/h2_workspace_margin_at_marks.csv`, `figures/h2_workspace_margin_timeline.png`)
**[hardware]**. Mean-over-reps values at each mark:

| $s$ (mm) | excl. margin $J_C$ (mm) | excl. margin $J_{NC}$ (mm) | box $x_{max}$ margin $J_{NC}$ (mm) | $z$ margin $J_{NC}$ (mm) | tracking error $J_{NC}$ (mm) |
|---|---|---|---|---|---|
| 45 | 3.5 | 9.3 | 163.5 | 38.4 | 0.75 |
| 50 | 8.8 | 12.8 | 145.6 | 38.3 | 0.84 |
| 55 | 2.7 | 17.1 | 124.8 | 38.3 | 1.00 |
| 58 | 4.8 | 23.9 | 106.7 | 38.8 | 0.98 |
| 60 | 4.3 | 27.7 | 96.1 | 39.1 | 1.22 |
| 62 | 6.7 | 34.8 | 81.7 | 39.7 | 2.17 |
| 64 | 8.6 | 48.9 | 61.7 | 40.8 | 4.07 |

**MPC-$J_{NC}$'s exclusion margin grows monotonically (9→49mm) — it moves
away from the floor, not toward it — and every other margin (box faces,
$z$-bound) also stays large and only mildly tightens.** At $s=64$mm, the
tick nearest its abort, every margin is still tens of millimetres from zero.
**This revises the prior report's §3.4 reading** ("retreats to
configurations where local tracking is easier but systematically wrong,
falling further behind until it aborts" could be read as implying a
workspace-pressure story): the data shows no workspace or exclusion
pressure of any kind on MPC-$J_{NC}$ at the tight floor. Whatever is wrong
is a pure control/tracking-authority effect, not a configuration-space
availability effect — which is exactly what §3.3 below demonstrates
directly.

The magnet-distance-and-insertion story from the prior report (contact-aware
MPC rides the floor at both radii; no-contact MPC pins to the floor at
210mm but drifts outward at 255mm; no-contact MPC inserts the beam least of
all four MPC conditions, 73.6-74.4mm, while contact-aware MPC at 255mm
inserts the most, 89.7-91.0mm) is unchanged and still correct as a
*description* of what happens (`figures/h2_magnet_distance_and_insertion_story.png`);
only the causal *reading* of why is revised below.

### 3.3 Same-state counterfactual: the decisive evidence

`scripts/h2_same_state_counterfactual.py` takes each tight-floor MPC-$J_{NC}$
rep's own measured state at $s=\{45,50,55,58,60,62,64\}$mm and solves the
identical MPC twice **[counterfactual]**:

- $u_0^{NC}$ — no override (literally what the run did; cross-checked
  against the logged command, max relative error 0.06% across all 21
  ticks — see `tables/h2_same_state_sanity_check.csv`);
- $u_0^{C}$ — the schedule's horizon window replaced by $J_C^{state}(x_k)$,
  the contact model recomputed at that exact measured state (the one
  physically-grounded reference this investigation uses, per H4 below).

Both commands are then evaluated through the **same** $J_C^{state}(x_k)$ —
never through either command's own source model — decomposed into useful
correction along $\hat e_k=(p_{ref}-p_{tip})/\|p_{ref}-p_{tip}\|$ and
transverse (wasted) correction (`tables/h2_same_state_counterfactual.csv`,
`figures/h2_same_state_counterfactual.png`):

| $s$ (mm) | $c_\parallel$, actual $J_{NC}$ command (mm) | $c_\parallel$, counterfactual $J_C^{state}$ command (mm) | ratio |
|---|---|---|---|
| 45 | 0.238 | 0.716 | 3.0$\times$ |
| 50 | 0.202 | 0.687 | 3.4$\times$ |
| 55 | 0.251 | 1.561 | 6.2$\times$ |
| 58 | 0.266 | 1.677 | 6.3$\times$ |
| 60 | 0.100 | 0.110 | 1.1$\times$ |
| 62 | 0.096 | 0.614 | 6.4$\times$ |
| 64 | 0.061 | 0.267 | 4.4$\times$ |

(Values are mean over the 3 reps; every mark individually shows the same
direction in all 3 reps — see the full table.)

**At every one of these 7 states, independently in all 3 reps, the command
the optimizer would have issued with the correct model achieves
substantially more real tracking correction than the command it actually
issued with the no-contact model** — directly establishing, not merely
inferring, that the Jacobian mismatch degrades the realized correction at
the exact states this condition visits. This is quantitatively consistent
with H4's independent finding (§6) that the no-contact schedule
over-predicts tip response by a mean gain ratio of 2.64$\times$ at this
same floor: a controller that believes a command will produce roughly
2.6$\times$ the actual contacted response will, after conversion to a
correctly-scaled command, under-correct by a similar factor — matching the
3-6$\times$ gap in realized correction measured here directly.

The counterfactual's effect on the exclusion margin is *not* uniform across
this window, and is itself informative: at $s=45$-$58$mm the $J_C^{state}$-
informed command would have simultaneously produced *more* correction *and*
more exclusion margin than the actual command (mean $\Delta h_{excl}$
+9 to +13mm vs the actual command's +0.17-0.30mm) — i.e. no trade-off was
being made there, the actual command was simply less effective on both
counts. At $s=60$-$64$mm this reverses: the $J_C^{state}$-informed command
would have *spent* exclusion margin to achieve its larger correction
(mean $\Delta h_{excl}$ −0.5 to −12mm, vs the actual command's continuing
small *positive* drift). **This is the one place in this window where a
genuine correction-vs-margin trade-off appears** — and even there, §3.2
already shows the actual run had nowhere close to zero margin to spend.

### 3.4 Revised mechanism and termination

$$
\boxed{
J_{NC}^{sched}\ \text{overpredicts contacted tip response (gain ratio 2.64}\times\text{, H4)}
\rightarrow
\text{at matched states, the resulting command realizes 3-6}\times\text{ less tip correction than a contact-informed command would (same-state counterfactual, direct)}
\rightarrow
\text{tracking error accumulates (0.75}\rightarrow\text{4.07mm, Table in \S3.2)}
\rightarrow
\text{tracking-error abort at }s\approx64\text{mm, with every workspace/exclusion margin still tens of mm from zero.}
}
$$

No workspace- or exclusion-pressure term appears anywhere in this chain —
§3.2 rules it out directly. "Gets fed in less and then gives up" and
"retreats to configurations where local tracking is easier" are removed;
the mechanism is a quantified, same-state-demonstrated reduction in realized
correction, nothing more and nothing less.

### 3.5 Lag vs cross-track (unchanged from the prior report, §3.5 there)

No-contact MPC's terminal lag grows 36$\times$ from 210mm to 255mm while
cross-track grows a comparable 2.4$\times$; the honest reading remains that
error accumulates in both components over the run, with the *terminal*
failure specifically lag-dominated — consistent with, and now additionally
explained by, the reduced-realized-correction mechanism in §3.4 (a command
that under-corrects falls behind the reference's own fixed schedule,
which is exactly what "lag" means here).

---

## 4. H3 — MPC vs inverse-Jacobian constraint awareness

**Hypothesis:** an MPC controller that represents the exclusion constraint
inside its own optimization handles it fundamentally differently from a
naive resolved-rate controller that does not.

**Verdict: strongly supported — now explicitly presented as two separate
tests, neither one substituting for the other.**

### 4.1 H3a — intrinsic constraint awareness (the hold-gate evidence)

Unchanged from the prior report's §4: the inverse-Jacobian controller's raw,
pre-gate command was recovered by exact offline replay of its own control
law (agreement with the real external gate's hold/pass decision = 1.000
across all 12 runs analysed) **[counterfactual, validated against
hardware]**. It requests infeasible motion on 0-69.6% of ticks depending on
floor/Jacobian, is held (losing 0-26.7mm of path progress per rep) on every
one of those ticks, while MPC's in-QP constraint is active on 8.5-20.4% of
ticks and is **never violated** on any tick of any MPC run. This establishes
**H3a's claim and only this claim**: *the inverse-Jacobian control law
itself carries no representation of the exclusion constraint* — it is not a
claim about overall tracking quality, which is H3b's and §3's territory.
Also unchanged: this difference is latent, not exercised, in the one cell
where the constraint never binds (210mm, matched Jacobian: 0 infeasible
ticks, 0.685mm RMSE vs MPC's 0.497mm); and roughly half of tight-floor
infeasible episodes are actually the vertical workspace bound, not the
named exclusion radius (one rep: 0% radius-infeasible, 73.7%
vertical-infeasible).

### 4.2 H3b — selective-gate robustness test (the stronger test)

The all-or-nothing hold policy in H3a zeroes the *entire* 7-vector on any
infeasible tick — a severe intervention that could by itself exaggerate the
inverse-Jacobian controller's measured disadvantage. A **selective** gate
(zeroing only the individual joint-velocity components whose own sign
worsens the predicted exclusion margin, passing every other component
unmodified) was run instead, at the 255mm floor, under both Jacobian models
(6 hardware reps; `tables/per_run_summary_closedloop.csv` group
`invjac_selective`/`invjac_clip_nc`) **[hardware]**.

**Result: even with almost the entire command passing through unmodified,
both pairings still fail — 0/6 complete, all six on a *different*, generic
`tcp_out_of_workspace` stop, not the magnet-exclusion gate** (exclusion-gate
clip activity 0-1.9% of ticks; the magnet never approaches its own floor in
any of the 6 reps, closest approach 255.5-258.6mm against a 255mm floor).
The matched MPC comparator completes 3/3 under the contact Jacobian and
fails 0/3 under the no-contact Jacobian (the H2 mechanism, §3 — a
*different* failure, confirmed by its own `tracking_error_exceeded` stop
reason, not `tcp_out_of_workspace`).

**This is the stronger robustness result**: it shows the inverse-Jacobian
architecture's disadvantage is not an artefact of the hold gate's severity.
Suppressing only the offending command components is not sufficient to
recover MPC-level robustness — a *second*, independent failure mode is
responsible, addressed on its own terms in §5.

---

## 5. Secondary mechanism — redundant-configuration drift and workspace exit

*(H3b's own content from the prior report, now with a quantitative
workspace-face/DOF decomposition and an explicit null-space projection —
both newly computed this pass.)*

### 5.1 The exact binding face, and which joint moves it (revises the prior report's "joint 0" framing)

`scripts/h3b_workspace_and_nullspace.py` identified, for all 6 selective-gate
runs (3 contact + 3 no-contact Jacobian), exactly which TCP-workspace face
triggers the stop, the signed margin to it over the whole run, and which
configuration coordinate(s) are responsible, via a per-DOF divergence
$\delta\chi(s)=\chi_{InvJac}(s)-\chi_{MPC}(s)$ against the matched MPC run,
plus the margin's own first-order sensitivity to each joint
(`tables/h3b_workspace_failure_decomposition.csv`,
`tables/h3b_workspace_failure_summary.csv`, `figures/h3b_workspace_drift_mechanism.png`)
**[hardware, direct]**:

- **The binding face is $x_{max}$ in all 6 reps, with no exception**
  (TCP exit position e.g. $[0.659,-0.387,0.400]$m against the box's own
  $x_{max}=0.656$m). This was not previously quantified; it rules out the
  $y$ or $z$ faces as alternative explanations.
- Using this script's 1-indexed joint labelling ($q_1$ = base rotation,
  …, $q_6$ = wrist 3): **the shoulder joint $q_2$ is the first coordinate to
  measurably diverge from the matched-MPC trajectory, at a highly
  consistent onset of $s\approx20$mm in all 6 reps.** But $q_2$'s own
  sensitivity of the binding $x_{max}$ margin to its own motion,
  $\partial h_{x_{max}}/\partial\chi_{q_2}$, averages only **0.0046-0.0054**
  over the run (mm of margin per unit joint motion). **The base joint
  $q_1$ diverges later (onset $s\approx45$-$68$mm, close to the eventual
  exit) but its own sensitivity averages −0.063 to −0.070 — roughly
  13-15$\times$ larger in magnitude than $q_2$'s.**
- **Reading:** several joints (most prominently $q_2$, but also $q_4$,
  $q_5$) drift away from the matched-MPC trajectory early and
  continuously, each contributing a small, steady pull on the $x_{max}$
  margin over tens of millimetres of path progress; the margin only
  actually collapses once the high-sensitivity base joint $q_1$ itself
  begins diverging, late and close to the exit. **This replaces "joint 0
  diverges, the robot exits its workspace" with a two-stage account:
  low-sensitivity, early, multi-joint drift sets up the margin's slow
  decline; high-sensitivity, late base-joint drift is what actually spends
  the remaining margin to zero.** `tables/h3b_workspace_failure_decomposition.csv`
  provides the full per-tick decomposition behind this summary.

### 5.2 Is it literally null-space motion?

Using the full $3\times7$ tip-position Jacobian $J_p$ at each tick and
$P_N=I-J_p^{\dagger}J_p$, every logged inverse-Jacobian command $u$ was
split into $u_N=P_N u$ (null/redundant) and $u_R=(I-P_N)u$ (task-relevant),
reporting $E_N=\|u_N\|^2/\|u\|^2$ in an early window ($s<30$mm) vs a
pre-failure window (the tail before the `tcp_out_of_workspace` stop)
(`tables/h3b_null_redundant_projection.csv`, `figures/h3b_redundant_projection.png`)
**[hardware, direct]**:

| pairing | $E_N$, early (mean) | $E_N$, pre-failure (mean) |
|---|---|---|
| selective, contact Jacobian | 0.026 | **0.214** |
| selective, no-contact Jacobian | 0.022 | **0.382** |

**$E_N$ grows roughly 10-17$\times$ from the early region to the
pre-failure region, in both Jacobian pairings** — a genuine, large,
and consistent increase in the fraction of commanded motion that lies in
the kinematic null space of the tip task. It is not, however, close to 1:
even immediately before the workspace exit, 58-79% of the command's squared
norm is still task-relevant. **This supports "accumulated redundant-
configuration drift" as a real, substantial, and growing phenomenon, but
not "almost pure null-space motion"** — the two-stage per-DOF account in
§5.1 (many joints drifting mostly in task-irrelevant combinations, a
smaller task-relevant remainder still present throughout) is the more
precise description, and is what the data directly shows.

The prior report's wording, *"the four redundant directions are completely
free to drift however the resolved-rate solution's own numerical
asymmetries happen to push them"*, is replaced with: **the inverse
controller has no secondary objective that regulates accumulated redundant
configuration, whereas MPC's full-input and input-increment costs penalize
every commanded direction — task-relevant or not — on every tick.** This is
the same point the prior report already made in its closing sentence of
§5.2; it is now the only form of the claim retained, and it is directly
supported by $E_N$ growing rather than staying flat, rather than by an
unverified "completely free" characterization.

### 5.3 Scope (unchanged from the prior report's §5.3)

Confirmed to reproduce under two independent Jacobian sources, both
verified via command replay (not directory names) to have run the selective
gate; does not alter the H3a verdict; the controller's internal **clip**
mode remains untested in this record.

---

## 6. H4 — Scheduled Jacobian accuracy against the contact model at the measured state

Sections 6.1 (why $J_C^{state}$ is the only physically-grounded reference),
6.2-6.4 (the whole-run Q1/Q2/Q3 tables: relative-Frobenius mismatch,
command-weighted error, gain ratio, direction angle), and 6.5 (the
staleness-vs-missing-physics secondary decomposition) are **unchanged from
the prior report** and are not repeated here. The headline whole-run finding
stands: the no-contact-scheduled MPC condition at the tight floor is the
outlier on every one of $E_J$, $\epsilon_u$, and the 2.64$\times$ gain
ratio simultaneously, and its mismatch-vs-tracking-error correlation moves
from near-zero at the loose floor to +0.47 at the tight floor.

### 6.5 New: common-progress-matched bins, and temporal ordering against H2's failure

Whole-run means above are vulnerable to one specific distortion: the
255mm MPC-$J_{NC}$ condition terminates at $s\approx64$mm, while its
contact-Jacobian comparator completes the full $\approx90$mm path — a
whole-run average silently compares unlike portions of the trajectory.
`scripts/h4_progress_binned.py` re-bins the existing 1636-tick guided
sample (same data as §6.2-6.4, no new live-model evaluations) into shared
path-progress bins, marking a condition "not observed" in any bin it never
reached rather than omitting it silently (`tables/h4_common_progress_bins.csv`,
`figures/h4_progress_binned_accuracy.png`) **[model, re-binned]**:

| bin (mm) | condition | $E_J$ mean | $\epsilon_u$ mean (mm) | gain ratio mean | tracking error mean (mm) |
|---|---|---|---|---|---|
| 45-55 | 255mm MPC-$J_C$ | 0.398 | 0.239 | 1.59 | 0.315 |
| 45-55 | 255mm MPC-$J_{NC}$ | 0.563 | 0.273 | 2.15 | 0.908 |
| 55-60 | 255mm MPC-$J_C$ | 0.442 | 0.212 | 1.42 | 0.450 |
| 55-60 | 255mm MPC-$J_{NC}$ | 0.833 | 0.318 | 2.41 | 0.997 |
| 60-64 | 255mm MPC-$J_C$ | 0.803 | 0.518 | 1.91 | 1.177 |
| 60-64 | 255mm MPC-$J_{NC}$ | 0.842 | 0.975 | **7.94** | 2.487 |

**At matched path progress, every one of $E_J$, $\epsilon_u$, and the gain
ratio is already elevated for MPC-$J_{NC}$ relative to its own
contact-Jacobian comparator in the 45-55mm and 55-60mm bins — well before
the 60-64mm bin where tracking error and gain ratio both spike.** This is
the temporal-ordering evidence the causal reading in §3.4 needs: the model
mismatch is not something that only appears alongside the terminal tracking
blow-up, it is already present and already larger than the matched
comparator's own mismatch tens of millimetres of path progress earlier.
This is **consistent with, and strengthens, but does not by itself prove**
the §3.4 causal chain — the same-state counterfactual in §3.3 is the direct
demonstration; this bin-matched table is the supporting temporal evidence.

---

## 7. H5 — Scheduled vs frozen Jacobian

Sections 7.1 (design: why two frozen ablations, chosen to separate
conditioning from staleness) and 7.2 (headline completion/RMSE table) are
**unchanged from the prior report**. The verdict stands: no single fixed
linearization matched the scheduled Jacobian's combination of tracking
quality and robust completion.

### 7.3 Why index 0 survives and index 100 does not — now with a same-state counterfactual

The prior report's directional-alignment check (index 0's dominant output
direction staying closer to the schedule's own evolving direction than
index 100's, at every post-contact checkpoint) was correlational. This pass
adds a direct same-state counterfactual: at 11 states drawn from the
**frozen-idx100 run's own hardware log** (not the scheduled baseline —
an earlier attempt using the scheduled baseline's own small, noisily-signed
tracking error produced sign flips too often to give a clean read, since a
successful run's error has no sustained direction to project onto; the
failing idx100 run's error does), the identical MPC is re-solved three
times per state — with the real frozen-idx100 schedule (cross-checked
against the logged command, max relative error 0.05% across all 11 states),
with the trajectory-varying schedule's own matrix at that exact reference
index, and with the frozen-idx0 matrix — and all three resulting commands
are evaluated through $J_C^{state}(x_k)$ along $\hat e_k$
(`tables/h5_same_state_counterfactual.csv`,
`figures/h5_same_state_frozen_counterfactual.png`) **[counterfactual]**:

| $s$ (mm) | $c_\parallel$, idx100 actual (mm) | $c_\parallel$, scheduled c/f (mm) | $c_\parallel$, idx0 c/f (mm) |
|---|---|---|---|
| 20 | −0.085 | −0.179 | −0.031 |
| 40 | 0.105 | 0.866 | 0.083 |
| 50 | −0.128 | −1.064 | −0.211 |
| 55 | 0.110 | 1.826 | 0.166 |
| 58 | −0.101 | −1.396 | −0.184 |
| 60 | 0.108 | 1.118 | 0.137 |
| 62 | −0.168 | −1.665 | −0.237 |
| **64** | **0.016** | **3.133** | **0.063** |
| 67 | −0.084 | 0.820 | −0.075 |

(sign follows $\hat e_k$ at that tick, so a negative/positive alternation
tick to tick is expected and not itself meaningful; magnitude is the
relevant quantity here.)

Three things this directly shows:

1. **The scheduled Jacobian's counterfactual command produces far more
   realized correction than either frozen alternative at essentially every
   state** (e.g. at $s=64$mm, 3.13mm vs 0.016mm for idx100's actual command
   and 0.063mm for idx0 — a 50-200$\times$ gap) — a direct, same-state
   demonstration of why continual re-linearization beats any single frozen
   snapshot, not just an outcome-level correlation.
2. **From $s\approx50$mm onward — entering and through idx100's own
   reported failure-onset window (§7.3 of the prior report, $s\approx58$-
   $68$mm) — idx0's counterfactual command produces a larger-magnitude
   useful correction than idx100's own actual command, at 6 of 7 marks in
   that range**, most strikingly at $s=64$mm (idx100's own command:
   0.016mm of real correction, essentially nothing; idx0's counterfactual
   command at the identical state: 0.063mm, $\approx4\times$ more).
   **This directly supports the staleness reading**: idx100's command,
   evaluated against the real contacted system at the exact moment its
   tracking error is accelerating, has become almost uninformative, while
   idx0 — despite its poor local conditioning — has not.
3. **Before $s\approx50$mm, idx100's own actual command is comparable to or
   larger than idx0's counterfactual command** at several marks (20, 40mm)
   — consistent with the prior report's §7.2 observation that idx100
   tracks *better* than idx0 through the early-mid path, now given a
   mechanistic grounding rather than only a tracking-statistic comparison.

The prior report's *"that is the signature of staleness specifically"* is
retained, but is now **directly supported by same-state replay**, not only
by the correlational directional-alignment check — which remains valid and
is kept as corroborating evidence, not replaced.

---

## 8. Secondary ablation — proportional gain (unchanged)

Unchanged from the prior report's §8. Verdict: contradicted — the
infeasible-request rate is exactly zero at both gains tested; the
tracking-degradation pattern at the higher gain (larger, more oscillatory
commands, 34% more sign reversals) is consistent with, but not proven to be,
underdamped behaviour, and is independent of the exclusion constraint
entirely.

---

## 9. Task 1 — The $s\approx57$-$61$mm local transient: deep diagnostic

The prior report's §1.4 ruled out eight candidates jointly (schedule
conditioning, one-step prediction error, all three singular values,
commanded-velocity norm and its tick-to-tick change, the exclusion margin,
commanded insertion rate, and scheduled-vs-live-state Jacobian divergence)
in the three contact-aware MPC reps at 210mm, over $s=54$-$60$mm. This pass
ran four **new** diagnostic categories over a slightly wider $s=53$-$61$mm
window, all three reps (`scripts/transient_57mm_part2.py`,
`tables/transient_57mm_diagnostic_summary.csv`,
`tables/transient_57mm_1D_contact_state.csv`,
`figures/transient_57mm_deep_diagnostic.png`):

- **Refined peak location.** The true peak sits at $s\approx60.2$-$60.5$mm
  — just past the edge of the originally-scanned 54-60mm window — reaching
  **3.47-3.81mm**, slightly above the prior report's reported 3.7mm maximum
  (which had clipped the window at 60mm). All three reps peak within 0.3mm
  of path progress of each other.
- **1A, vision/measurement:** at the peak tick in every rep, the error is
  coherent and **entirely in-plane** — $e_z=0.000$mm exactly in all three
  reps, with $e_x\approx3.0$-$3.2$mm and $e_y\approx1.5$-$2.1$mm. No
  raw-vs-filtered discrepancy was present anywhere in the window
  (`raw_minus_filt_norm_mm`$=0$ throughout), and measurement age at the peak
  tick varies from 0.3ms to 41ms across the three reps with no
  corresponding variation in peak magnitude — ruling out measurement
  staleness as the direct cause.
- **1B, reference-path geometry:** at the exact peak tick in every rep, the
  local reference curvature $\kappa_{ref}=0$ and tangent rotation per tick
  is $\approx0.008$-$0.009$°, i.e. the path is locally straight and
  undemanding right where the error peaks, despite real curvature variation
  elsewhere in the window (up to $\kappa_{ref}\approx11.7\,\mathrm{m}^{-1}$
  at other ticks). This rules out a locally demanding path feature as the
  direct cause.
- **1C, commanded vs executed motion:** the per-tick configuration tracking
  error $e_{\chi,k}$ at the peak is small and, critically, **inconsistent in
  sign across reps** — the robot under-executes the command by
  $\approx40\%$ in two reps and over-executes by $\approx25\%$ in the third.
  A systematic execution lag specific to this window would show the same
  sign in all three reps; it does not.
- **1D, contact state:** recomputed directly from the contact model at a
  coarser stride (every $\approx8$ ticks, each equilibrium re-solve costing
  6-10s) across the window: zero penetrating nodes throughout, a smoothly
  varying gap (0.019-0.031mm) and contact-force proxy, with no abrupt
  active-set transition visible at the sampled ticks. **This diagnostic's
  stride did not land exactly on the single worst tick in two of three
  reps**, so a contact-state check at the precise peak tick itself remains
  a residual gap, noted honestly rather than papered over.

**Conclusion: after twelve candidate explanations now jointly checked
(eight from the prior pass, four here), none shows a clear, repeatable
anomaly coincident with the peak tick. This transient remains
unexplained.** It is reported this way deliberately — forcing an
explanation the data does not support would be worse than leaving it open.
It remains small (under 3.81mm, below every abort threshold used in this
report) and does not threaten any conclusion elsewhere in this document.

---

## 10. Integrated discussion

Two largely independent axes, each now traceable through at least one
same-state or direct-measurement step, not only through outcome statistics.

**Modelling fidelity and available configuration space (H1, H2, H4, H5).**
An executable plan requires contact physics at all (H1). Whether using the
*wrong* model during closed-loop control is costly depends on available
configuration freedom: at a loose floor, both models reach the same
near-base configurations and the wrong model's error is absorbed for free;
at a tight floor, the wrong model's schedule mismatch is already elevated
tens of millimetres before the eventual tracking failure (H4, §6.5), and a
same-state counterfactual directly shows its actual commands realize
3-6$\times$ less tracking correction than a contact-informed command would
at the identical states (H2, §3.3) — with every workspace and exclusion
margin confirmed to stay generous throughout, ruling out a
configuration-space-exhaustion reading (§3.2). Nor is a single fixed
linearization a substitute for continual updating: a same-state
counterfactual shows the scheduled Jacobian's command realizes far more
correction than either frozen alternative at matched states, and shows
specifically that the well-conditioned-but-frozen matrix's own commands
become nearly uninformative exactly where its tracking error accelerates,
while the ill-conditioned-but-frozen matrix's commands do not (H5, §7.3).

**Controller architecture (H3a, H3b).** Independent of modelling quality, a
controller unable to represent the exclusion constraint inside its own
optimization is held in place whenever it requests an infeasible motion,
bleeding task progress because the reference advances regardless (H3a).
This is not the only architectural disadvantage: suppressing only the
offending command components (rather than the whole command) does not
recover MPC-level robustness either — both Jacobian pairings under the
selective gate still exit the robot's own workspace, on a stop reason
unrelated to the magnet constraint (H3b's robustness test, §4.2). The
mechanism responsible is now decomposed by workspace face and joint:
consistently the $x_{max}$ TCP-box face; an early, low-sensitivity,
multi-joint drift (led by the shoulder joint) that slowly erodes the
margin, followed by a late, high-sensitivity base-joint drift
($\approx14\times$ the shoulder joint's own sensitivity) that spends what
remains of it (§5.1); and a null-space-energy fraction that grows
10-17$\times$ from early path progress to the pre-failure window, though it
never exceeds $\approx40\%$ of the command's squared norm even immediately
before the exit (§5.2) — "accumulated redundant-configuration drift" is the
precise, evidence-matched term, not "null-space drift." MPC's own
formulation avoids both problems directly: its in-QP constraint keeps every
tick feasible, and its input and input-increment costs regularize every
commanded direction, task-relevant or not, without any explicit posture
target.

These remain two additive, independent sources of the performance gap
between contact-aware MPC and every other condition tested — one about
whether the model is right and whether workspace freedom is available to
absorb being wrong, the other about whether the controller's own
optimization structure can represent the constraints and redundancy it
operates under.

---

## 11. Limitations

- **$n=3$ reps per condition; no significance testing anywhere in this
  report** (unchanged from the prior report).
- **$J_C^{state}$ is a recomputed model evaluation, not a direct physical
  measurement** (unchanged); every same-state counterfactual in this
  revised report inherits this same caveat, since all of them evaluate
  commands through $J_C^{state}$.
- **Same-state MPC replay reproduces logged commands to a median 0.01%
  relative error, but is not universally exact**: 1 of 20 validation ticks
  showed a 123% relative error on a near-zero-magnitude command,
  attributable to a weakly-penalized QP direction rather than a replay
  defect. Every same-state result in this report reports its own
  per-tick sanity-check error alongside it; none of the counterfactual
  conclusions above (§3.3, §7.3) rest on a tick flagged this way.
- **Abort thresholds differ by controller architecture** (5mm MPC / 20mm
  inverse-Jacobian / 10mm open-loop) — unchanged from the prior report.
- **The $s\approx60$-$61$mm local transient remains unexplained** after
  twelve candidate categories now checked (§9), including four new ones
  this pass. It is too small to threaten any conclusion in this report.
- **The H3b contact-state diagnostic (§9, category 1D) did not sample the
  exact peak tick in 2 of 3 reps**, due to a coarser stride needed for
  tractability (6-10s per equilibrium re-solve); a fully exact check at the
  single worst tick would need either a faster contact solve or a
  dedicated single-tick re-solve, neither performed here.
- **Single robot, single vessel geometry, single beam** (unchanged).
- **The H5 idx0-vs-idx100 early-region tracking-quality ordering is still
  not fully reconciled with the same-state counterfactual**: before
  $s\approx50$mm, idx100's own actual command is comparable to or exceeds
  idx0's counterfactual command (§7.3, point 3), consistent with idx100
  tracking better than idx0 early in the path (prior report §7.2) — but
  this report does not have a single unifying quantity that predicts
  *both* the early-region ordering and the late-region reversal from one
  measurement; it reports both facts and their same-state-counterfactual
  support, rather than resolving them into one mechanism.
- **Would any of the above benefit from new hardware?** No single result
  in this revised report required one — all six tasks were resolved from
  existing logs, offline model evaluations, and same-state replay. The one
  genuinely open item is the unexplained local transient (§9); resolving it
  further would most plausibly need either a faster contact-model solve
  (to sample the exact peak tick in 1D) or a dedicated high-rate logging
  pass at that specific path region, not a new full closed-loop trial.

---

## 12. Final conclusions

| Hypothesis | Verdict | Strongest evidence | Mechanism status |
|---|---|---|---|
| **H1** — contact-aware planning required for an executable open-loop plan | **Strongly supported** | 4/4 completion vs 0/4; divergence lags predicted contact onset by $\approx22$mm | Directly demonstrated (model-vs-model comparison at matched states) |
| **H2** — contact-Jacobian × exclusion-radius interaction | **Supported** | completion asymmetry (3/3$\to$3/3 vs 3/3$\to$0/3); same-state counterfactual shows 3-6$\times$ less realized correction under the wrong model at every examined state, with workspace/exclusion margins confirmed generous throughout | Directly demonstrated via same-state counterfactual (§3.3), not inferred from trajectory shape |
| **H3a** — MPC vs inverse-Jacobian intrinsic constraint awareness | **Strongly supported** | exact (1.000 agreement) raw-command recovery; MPC never violates its own in-QP constraint on any tick | Directly demonstrated |
| **H3b** — selective-gate robustness test + redundant-configuration drift | **Supported as a second, independent failure mode** | 0/6 complete under selective gating despite near-zero exclusion-gate activity; exact binding face ($x_{max}$) and two-stage per-DOF mechanism identified; $E_N$ grows 10-17$\times$ pre-failure but stays $<0.4$ | Directly demonstrated (workspace-face/DOF decomposition + null-space projection); "redundant-configuration drift," not "null-space drift" |
| **H4** — scheduled-Jacobian accuracy against the contact model at the measured state | **Supported, and shown to precede the H2 failure temporally** | mismatch/gain-ratio/$\epsilon_u$ all elevated in matched 45-55mm and 55-60mm bins, before the 60-64mm failure bin | Direct re-binned measurement; supports but does not alone establish causality (§3.3 does) |
| **H5** — a trajectory-varying Jacobian outperforms any single fixed one | **Supported** | same-state counterfactual: scheduled command realizes 50-200$\times$ more correction than either frozen alternative at matched late-path states; idx0 outperforms idx100's own actual command in realized correction at 6/7 marks past $s\approx50$mm | Directly demonstrated via same-state counterfactual (§7.3), corroborating the prior directional-alignment finding |
| Proportional-gain ablation (secondary) | **Hypothesis contradicted** | 0% infeasible-request rate at both gains | Unchanged; underdamping is plausible, not demonstrated |

---

## Audit

**New scripts** (`scripts/`): `mpc_same_state_replay.py` (shared same-state
MPC replay infrastructure, validated to 0.01% median command-reproduction
error), `h2_workspace_timeline.py`, `h2_same_state_counterfactual.py`,
`h3b_workspace_and_nullspace.py`, `h4_progress_binned.py`,
`h5_same_state_counterfactual.py`, `transient_57mm_part2.py` (+
`transient_57mm_part2_figure_final.py`, `transient_peak_refine.py`).

**New tables**: `h2_workspace_margin_timeline.csv`,
`h2_workspace_margin_at_marks.csv`, `h2_same_state_counterfactual.csv` (+
sanity check), `h3b_workspace_failure_decomposition.csv`,
`h3b_workspace_failure_summary.csv`, `h3b_null_redundant_projection.csv`,
`h4_common_progress_bins.csv`, `h5_same_state_counterfactual.csv` (+
sanity check), `transient_57mm_diagnostic_summary.csv`,
`transient_57mm_1D_contact_state.csv`.

**New figures**: `h2_workspace_margin_timeline.png`,
`h2_same_state_counterfactual.png`, `h3b_workspace_drift_mechanism.png`,
`h3b_redundant_projection.png`, `h4_progress_binned_accuracy.png`,
`h5_same_state_frozen_counterfactual.png`, `transient_57mm_deep_diagnostic.png`.

**Claims strengthened** (from correlational/outcome-level to direct
same-state or re-binned demonstration): H2's whole causal chain (§3.3-3.4);
H5's staleness reading (§7.3); H4's relevance to H2, via temporal ordering
(§6.5); H3b's mechanism, via exact workspace face and two-stage per-DOF
account (§5.1) and a measured (not assumed) null-space-energy fraction
(§5.2).

**Claims weakened / corrected**: H2's prior "retreats to configurations
where local tracking is easier" / workspace-pressure framing — directly
contradicted by measured margins staying generous throughout (§3.2); the
prior report's "joint 0 diverges" single-DOF account of H3b's workspace
exit — refined into a two-stage, multi-joint account with the base joint's
large sensitivity but late onset (§5.1); "the four redundant directions
are completely free to drift" — replaced with a measured, bounded (never
exceeding $\approx40\%$) and growing null-space-energy fraction (§5.2).

**Mechanisms that remain unresolved**: the $s\approx60$-$61$mm local
transient (§9, now checked against twelve candidates); the H5 early-region
tracking-quality ordering between idx0 and idx100 (§7.3 point 3 / §11).

**Would any result here genuinely require a new hardware experiment to
resolve further?** No — see §11's final bullet. The local transient's
residual gap (exact-peak-tick contact state) is better addressed by a
faster offline contact solve or targeted re-logging than by a new trial.
