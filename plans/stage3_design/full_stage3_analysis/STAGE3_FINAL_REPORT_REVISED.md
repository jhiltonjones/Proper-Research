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

**Revision note (second pass).** An external review of the first version of
this document identified several further issues, addressed in place rather
than as an appendix: (1) §3.2's "pure control/tracking-authority effect, not
a configuration-space availability effect" over-corrected the first pass's
own fix and is softened; (2) §3.3/§3.4 and §7.3's same-state counterfactuals
both originally scored commands with a metric
($c_\parallel=\hat e_k^T J\Delta\chi$) whose sign was discarded despite being
physically meaningful — correcting this **reverses** both subsections'
headline "X times more correction" claims rather than merely softening
them, and both are now reported as directly-established-on-magnitude-only,
with the directional claim downgraded to attempted-and-inconclusive; (3) H2's
§3.3 counterfactual is clarified to have substituted $J_C^{state}$ repeated
across the whole horizon, and a second, literal contact-schedule-window
counterfactual is added alongside it; (4) §5.2's null-space finding was
*at the time* reported as strengthened by a pipeline-stage decomposition
(this specific point was itself subsequently found to rest on a flawed
premise — see the next revision note below, which supersedes it); (5) §6.5's
gain-ratio figures are checked for denominator robustness; (6) §9's
transient diagnostic is extended with an exact-peak-tick same-state replay,
which **reverses its own "remains unexplained" verdict** for two of three
reps; (7) the integrated discussion's claim that MPC's cost structure
"directly avoids" the redundant-drift failure mode is softened to
"consistent with," since no ablation isolates those cost terms as causal;
(8) three numerical/terminology slips are corrected (H1's completion count,
H5's "any single fixed" overclaim, and an $s$-vs-insertion-length mix-up in
§6.5). Every change is made in place, at the point it occurs, not collected
here — this note is a map of where to look, not a substitute for reading
those sections.

**Revision note (§5.2 correction, most important single correction in this
document).** A later reviewer identified a mathematical inconsistency in
§5.2's null-space-energy analysis: a damped-least-squares command is, for
any damping value, guaranteed to lie exactly in the row space of the
Jacobian it was solved with, so projecting it onto the null space of that
**same** Jacobian must give (near-)zero energy — not the $0.2$-$0.4$ the
original analysis reported. Direct verification confirmed the original
analysis had projected the command (solved with the **scheduled**
Jacobian) onto the null space of the **live, state-recomputed** contact
model — a different matrix. $E_N$ was therefore never a measurement of
"unregulated null-space motion in the control law's own solve" (a quantity
that is identically zero by construction and so could never have shown
this); it is a measurement of schedule-vs-state Jacobian mismatch,
channelled through the executed command. §5.2 is rewritten in place to
report this correction in full, including the verification numbers, and
every downstream reference to "null-space energy" in §5.4, §10, §12, and
this audit is updated to describe $E_N$ correctly. The underlying *data*
($E_N$'s numerical values, its growth, its correlation with margin collapse)
is unchanged and still reported; only the characterization of what it
measures is corrected. "Accumulated redundant-configuration drift" as a
phenomenon is **not** withdrawn — it now rests on §5.1's direct joint-angle
measurement alone, which does not depend on this issue.

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

**Verdict: supported, on the completion asymmetry and the magnitude-level
evidence (workspace margins, H4's gain-ratio mismatch, and the same-state
command-magnitude difference). The specific claim that the wrong model's
command is directionally *less effective* at matched states was attempted
via a same-state counterfactual and found inconclusive — see §3.3 — not
directly demonstrated as an earlier version of this section claimed.**

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
pressure of any kind on MPC-$J_{NC}$ at the tight floor.

This needs to be stated precisely, because it is narrower than "configuration
space doesn't matter here" — H2's own hypothesis, and §3.1's loose-vs-tight
floor comparison, is that the exclusion floor's tightness is exactly what
determines whether the wrong model's error is absorbable. What this
subsection establishes is specifically that **the failure is not caused by
the controller reaching a workspace or exclusion boundary** — not that the
tighter floor is irrelevant to the failure. The tighter feasible set is very
plausibly still *why* the Jacobian error becomes consequential in the first
place (a loose feasible set lets the correct-model and wrong-model
controllers settle on similar, equally-effective configurations, per
§3.1/§3.4's $\Delta_J$ sign flip between floors); it just does not do so by
being exhausted. Stated this way: **rather than the controller reaching a
workspace or exclusion boundary, under the tighter feasible set the
no-contact model produces insufficiently effective corrections despite
substantial remaining constraint slack** — which is exactly what §3.3 below
demonstrates directly.

The magnet-distance-and-insertion story from the prior report (contact-aware
MPC rides the floor at both radii; no-contact MPC pins to the floor at
210mm but drifts outward at 255mm; no-contact MPC inserts the beam least of
all four MPC conditions, 73.6-74.4mm, while contact-aware MPC at 255mm
inserts the most, 89.7-91.0mm) is unchanged and still correct as a
*description* of what happens (`figures/h2_magnet_distance_and_insertion_story.png`);
only the causal *reading* of why is revised below.

A direct, geometry-grounded view of the same contrast — the **beam tip's**
own measured trajectory (not the magnet's) plotted against the vessel's
actual centreline and wall boundaries, reconstructed from the lumen
geometry file rather than inferred — makes the completion asymmetry
visually immediate (`scripts/tip_vs_vessel_geometry_255mm.py`,
`tables/vessel_geometry_xy_mm.csv`, `tables/tip_vs_vessel_geometry_255mm.csv`,
`figures/tip_vs_vessel_geometry_255mm.png`) **[hardware + geometry, direct]**:
both conditions' tips hug the inner wall through the bend in close
agreement with each other and with each controller's own offline reference;
MPC-$J_{NC}$'s three reps all stop abruptly partway up the post-bend
straight segment (marked with $\times$), while MPC-$J_C$'s three continue
to the end of that segment and complete. The two conditions' own offline
references (each condition's own plan, contact-aware vs no-contact) are
shown lightly and are close to each other and to the wall — consistent
with §2's finding that the two models agree well before contact and
diverge after it.

### 3.3 Same-state counterfactual: what it establishes directly, and what it does not

`scripts/h2_same_state_counterfactual.py` takes each tight-floor MPC-$J_{NC}$
rep's own measured state at $s=\{45,50,55,58,60,62,64\}$mm and solves the
identical MPC twice **[counterfactual]**:

- $u_0^{NC}$ — no override (literally what the run did; cross-checked
  against the logged command, max relative error 0.06% across all 21
  ticks — see `tables/h2_same_state_sanity_check.csv`);
- $u_0^{C,state}$ — the schedule's horizon window replaced by
  $J_C^{state}(x_k)$, the contact model recomputed once at the exact
  measured state and held constant across the horizon.

A follow-up pass (`scripts/h2_signed_and_schedule_counterfactual.py`) added a
second, "controller-realistic" counterfactual, $u_0^{C,sched}$, that
substitutes the real contact schedule's own per-horizon-step window at that
exact reference index — a genuinely different command from $u_0^{C,state}$
($\|u_0^{C,state}-u_0^{C,sched}\|$ relative to either command's own norm
grows from $\approx0.2$-$0.4$ at $s=45$-$58$mm to $\approx1.0$-$2.0$ at
$s=60$-$64$mm — these two counterfactuals increasingly diverge from each
other exactly in the late-path region this section cares about, so neither
can stand in for the other there).

**What is directly, robustly established: the actual command differs
substantially from either counterfactual at every one of the 7 states, in
every rep** — $\|u_0^{NC}-u_0^{C,state}\|$ and $\|u_0^{NC}-u_0^{C,sched}\|$
are both comparable in size to the commands' own norms throughout
(`tables/h2_signed_and_schedule_counterfactual.csv`). This magnitude
comparison does not depend on any choice of sign convention and is not in
question.

**What is not established: which command would actually have tracked
better.** The original version of this subsection converted both commands'
predicted tip motion into a scalar "useful correction,"
$c_\parallel=\hat e_k^T J_C^{state}\Delta\chi$ with
$\hat e_k=(p_{ref}-p_{tip})/\|p_{ref}-p_{tip}\|$, and reported the
$J_C^{state}$-informed command as realizing "3-6$\times$ more correction." An
external review correctly identified the same defect in this metric that
affects H5's §7.3: discarding $c_\parallel$'s sign (as the original text did)
throws away the one piece of information that distinguishes a genuinely
useful command from a harmful one, and MPC's command is not intended to
correct only the *current* tick's error in the first place — it is solved
over a multi-step, delay-compensated horizon. Replacing $c_\parallel$ with
the same sign-respecting, one-step-ahead $\Delta e$ metric used in §7.3
(evaluated via $p_{ref,k+1}$, through $J_C^{state}$, for all three command
variants) **reverses the original claim rather than confirming it**: the
*actual* $J_{NC}$ command shows $\Delta e>0$ (a small, real predicted
improvement) at all 21 of 21 (state, rep) combinations, while the
$J_C^{sched}$-window counterfactual is actually *worse* than the actual
command at all 21/21, and the $J_C^{state}$-repeated counterfactual is worse
at 13/21.

**This reversal is not read here as "the no-contact command was actually
fine" — that would simply trade one metric-driven overclaim for its
opposite.** The same delay/horizon mismatch that makes the one-step metric
untrustworthy for the scheduled command in §7.3 applies here too: the
$J_C$-informed counterfactual commands are, if anything, larger in
magnitude and more sign-variable tick to tick than the actual command, which
is exactly the pattern a horizon-optimizing command directed at a future
rather than immediate state would produce. Neither the discarded-sign
original metric nor this corrected one-step metric can be trusted to settle
which command is actually better for a delay-aware, multi-step optimizer; a
real resolution needs the controller's own multi-step predicted-cost change
($\Delta V_{track}$, propagated through its horizon and delay buffer), which
is not computed in this pass. **The same-state counterfactual mechanism for
H2 is therefore reported as attempted and inconclusive, exactly as for H5
(§7.3) — not as a directly demonstrated mechanism**, and the "3-6$\times$ gap
in realized correction" claim is withdrawn.

### 3.4 Revised mechanism and termination

What remains directly supported, independent of the inconclusive same-state
metric above:

$$
\boxed{
\begin{aligned}
&J_{NC}^{sched}\ \text{over-predicts contacted tip response by a magnitude ratio, independent of sign}\\
&\text{(gain ratio 2.64}\times\text{ whole-run mean, H4 \S6; already elevated before the terminal}\\
&\text{tracking blow-up, \S6.5 — both magnitude-only comparisons, unaffected by the sign issue above)}\\
&\downarrow\\
&\text{at matched states, the resulting command differs substantially in magnitude from what}\\
&\text{a contact-informed model would command (same-state counterfactual, direct; §3.3)}\\
&\downarrow\\
&\textit{[whether this makes the realized tracking better or worse, one step or several steps ahead,}\\
&\textit{is not established by any metric computed in this report — open, §11]}\\
&\downarrow\\
&\text{tracking error accumulates (0.75}\rightarrow\text{4.07mm, Table in \S3.2)}\\
&\downarrow\\
&\text{tracking-error abort at }s\approx64\text{mm, with every workspace/exclusion margin still tens of mm from zero (\S3.2).}
\end{aligned}
}
$$

No workspace- or exclusion-pressure term appears anywhere in this chain —
§3.2 rules it out directly, and that part of the chain is unaffected by the
metric problem above (it is a margin measurement, not a correction-direction
claim). "Gets fed in less and then gives up" and "retreats to configurations
where local tracking is easier" remain removed. What is added by this
pass's correction: the magnitude-level mismatch and margin findings are
retained as directly demonstrated; the *directional* same-state mechanism
step (whether the wrong model's command is actually less effective, not
merely different) is downgraded to attempted-and-inconclusive, matching the
same honest standard applied to H5.

### 3.5 Lag vs cross-track (unchanged from the prior report, §3.5 there)

No-contact MPC's terminal lag grows 36$\times$ from 210mm to 255mm while
cross-track grows a comparable 2.4$\times$; the honest reading remains that
error accumulates in both components over the run, with the *terminal*
failure specifically lag-dominated. This pattern (falling behind the
reference's own fixed schedule, rather than diverging laterally from it) is
*consistent with* a command that is directionally less effective than a
contact-informed one would be — but, per §3.3's revised conclusion, that
directional claim is not established, so this lag/cross-track pattern is
reported as a further direct observation, not as confirmation of the
reduced-correction mechanism.

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
workspace-face/DOF decomposition — newly computed this pass — and a
schedule-vs-state mismatch projection that was originally characterized as
a null-space projection of the control law's own behaviour; §5.2 reports
why that characterization was wrong and what the quantity actually measures.)*

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

### 5.2 Is it literally null-space motion? — no, and a direct check shows why the original framing was wrong

**This subsection was substantively corrected during a later pass, after an
external reviewer identified a mathematical inconsistency in the original
$E_N$ analysis. The correction is reported in full, including what was
wrong and why, rather than silently replacing the old numbers.**

**The problem, stated precisely.** For damped least squares,
$u=J^T(JJ^T+\lambda^2I)^{-1}e$, the command $u$ is — for *any* damping
$\lambda$ — a linear combination of the columns of $J^T$, and therefore lies
*exactly* in the row space of $J$. If $P_N=I-J^\dagger J$ is built from that
**same** $J$, then mathematically $P_Nu=0$ up to floating-point precision,
always, regardless of damping, conditioning, or how close $J$ is to
singular. A nonzero $E_N=\|P_Nu\|^2/\|u\|^2$ of $0.2$-$0.4$, as the original
version of this subsection reported, is therefore only possible if $P_N$ was
built from a **different** matrix than the one the command was actually
solved with.

**Direct verification, done immediately on this concern being raised.** The
controller's actual command (`proper_research/controllers/inverse_jacobian_
controller.py`, `solve()`, confirmed by reading the source directly) uses
the **scheduled** Jacobian at that tick's reference index,
$J_{sched}[\text{index}]$ — not a live recomputation. Every $E_N$ value in
the original version of this subsection, however, was computed with
$P_N$ built from `live_jac.live_jacobian`, i.e. $J_C^{state}(x_k)$ or
$J_{NC}^{cf}(x_k)$ — the contact model **recomputed at the measured state**,
a genuinely different matrix from $J_{sched}$ whenever the two disagree.
`scripts/verify_null_space_projection_jacobian.py` recomputes $E_N$ both
ways at three sample ticks of one run, confirming this exactly:

| tick | $E_N$ with $P_N$ from $J_{sched}$ (same $J$ as the solve) | $E_N$ with $P_N$ from $J_{live}$ (different $J$) | $\|J_{live}-J_{sched}\|/\|J_{sched}\|$ |
|---|---|---|---|
| 50 | $4.7\times10^{-10}$ | 0.047 | 0.145 |
| 250 | $8.5\times10^{-8}$ | 0.074 | 0.337 |
| 350 | $1.1\times10^{-4}$ | 0.587 | 0.342 |

**Using the same Jacobian the command was solved with gives $E_N\approx0$
to machine precision, exactly as the mathematics requires. All of the
reported $0.2$-$0.4$ signal came from using a different Jacobian for the
projection than for the solve.**

**What $E_N$, as actually computed, measures instead.** It is not, and
given the mathematics above *cannot be*, a measurement of "unregulated
null-space motion generated by the control law's own solve" — that quantity
is identically zero by construction for any damped-least-squares command,
checked against its own Jacobian. What $E_N$ actually measures is: **the
fraction of the executed (schedule-derived) command that would be
redundant if the contact model recomputed at the current measured state
were used instead of the schedule** — i.e. a null-space-projected view of
*schedule-vs-true-state Jacobian mismatch*, the same underlying quantity H4
(§6) measures via relative-Frobenius norm, command-weighted error, and gain
ratio, now viewed through a different (null-space) lens, channelled through
the direction of the actual executed command. Its growth from
$\approx0.02$-$0.03$ (early) to $\approx0.21$-$0.38$ (pre-failure), in both
Jacobian pairings, is therefore better read as **consistent with** H4's
independent finding that schedule-vs-state mismatch grows through the path
(especially post-contact) than as a new, independent mechanism of its own.

**What this means for "accumulated redundant-configuration drift."** The
*term* is retained, but its evidentiary basis changes: it now rests on
§5.1's direct, Jacobian-independent measurement (joint angles $q_1,q_2$
diverging from the matched-MPC trajectory over path progress — a
measurement that does not depend on which Jacobian is used for any
projection) rather than on this $E_N$ analysis. The previously-made claim
*"the inverse controller has no secondary objective that regulates
accumulated redundant configuration, whereas MPC's ... costs penalize every
commanded direction"* remains a defensible **interpretation** of §5.1's
joint-divergence evidence, but is no longer additionally supported by a
null-space-energy measurement, because that measurement was not measuring
what it was reported to measure.

**The pipeline-stage check (§5.2's own follow-up) is still informative, once
correctly captioned.** $E_N$ (in its actual, schedule-vs-state-mismatch
sense) was computed separately for $u_{raw}$ (pre-clip), $u_{clipped}$, and
$u_{gated}$ (the logged command) (`tables/h3b_pipeline_stage_nullspace.csv`,
`figures/h3b_pipeline_stage_nullspace.png`) **[model, direct]**:

| pairing | stage | $E_N$, early (mean) | $E_N$, pre-failure (mean) |
|---|---|---|---|
| selective, contact | $u_{raw}$ | 0.058 | 0.208 |
| selective, contact | $u_{clipped}$ | 0.058 | 0.214 |
| selective, contact | $u_{gated}$ | 0.058 | 0.214 |
| selective, no-contact | $u_{raw}$ | 0.023 | 0.381 |
| selective, no-contact | $u_{clipped}$ | 0.023 | 0.382 |
| selective, no-contact | $u_{gated}$ | 0.023 | 0.382 |

All three stages show the same growth, confirming the schedule-vs-state
mismatch signal already exists in the raw, pre-clip command and is not
injected or altered by the velocity/acceleration clip or the selective
gate — a valid and still-useful finding, but now correctly read as "clipping
and gating do not add or remove schedule-vs-state mismatch" rather than
"the growth is intrinsic to the control law's own null-space behaviour,"
since the latter was never a quantity this projection could measure in the
first place.

### 5.3 Scope (unchanged from the prior report's §5.3)

Confirmed to reproduce under two independent Jacobian sources, both
verified via command replay (not directory names) to have run the selective
gate; does not alter the H3a verdict; the controller's internal **clip**
mode remains untested in this record.

### 5.4 Origin of inverse-controller lag and redundant drift

§5.1-5.3 establish the *end* of a possible causal chain (workspace exit;
and, per §5.2's corrected reading, a growing schedule-vs-state Jacobian
mismatch channelled through the command, measured in two narrow windows).
They do not establish whether that end is preceded by a staged sequence —
$e_{lag}\uparrow\to\|u_{raw}\|\uparrow\to E_N\uparrow\to$ configuration
drift $\to h_{x_{max}}\downarrow\to$ exit — or whether a Jacobian-mismatch-
driven *reduction in effective tracking gain* is what starts that sequence
in the first place. This subsection tests both questions directly, for all
6 selective-gate runs, at full trajectory resolution (not just the two
narrow windows §5.2 used. **Note on $E_N$ below:** per §5.2's correction,
$E_N$ is a schedule-vs-state-mismatch quantity, not a measurement of the
control law's own null-space behaviour — it is used in this subsection for
exactly that reason, as a companion measure of mismatch alongside $e_{lag}$,
$k_\parallel$, and excess lag, not as independent evidence of "redundant
drift" in the control-law sense.)

**A critical constraint on any explanation, stated up front and honoured
throughout:** both pairings — contact-Jacobian and no-contact-Jacobian —
fail 0/3, on the *same* `tcp_out_of_workspace` / $x_{max}$ face (§5.1). Any
mechanism offered below for *why* lag or drift develops must not be read as
explaining the shared failure itself unless it operates equally under both
Jacobians; a mechanism that is specific to the no-contact Jacobian can only
ever be a contributor to one pairing's version of a failure that also,
independently, happens to the other.

`figures/lag_three_layer_synthesis.png` assembles the evidence this
subsection develops into a single three-row figure, built entirely from the
same data computed below (no separate analysis of its own) — one row per
layer of the proposed mechanism, both pairings and all 3 reps overlaid, with
contact onset ($s=28.5$mm) and each pairing's own mean exit $s$ marked as
vertical lines in every row:

- **Row 1 (Layer 1):** measured $e_{lag}$ for both pairings against the
  shared theoretical $K_p=0.6$ baseline $v_s\Delta t/K_p$ (black dashed) —
  both pairings track this shared floor closely before $s\approx40$mm.
- **Row 2 (Layer 2):** $k_\parallel$ for both pairings against the nominal
  $K_p=0.6$ (black dashed) — the no-contact pairing visibly sits below the
  contact pairing, and below nominal, from shortly after contact onset.
- **Row 3 (Layer 3):** $E_N$ — the schedule-vs-state mismatch energy
  channelled through the executed command, §5.2's corrected quantity, *not*
  a measurement of the control law's own null-space behaviour — (solid, left
  axis) rising alongside $h_{x_{max}}$ (dotted, right axis) falling toward
  zero at exit, both pairings.

The three rows make the full chain visible at a glance: Layer 1's shared
lag floor is visibly left behind by the no-contact pairing around
$s\approx40$-$50$mm, exactly where Layer 2 shows its $k_\parallel$ departing
from the contact pairing's own, and exactly where Layer 3's $E_N$ begins its
own rise alongside the shared-face exit. The per-quantity numbers and
statistics behind each row are developed in §5.4.1-§5.4.4 below.

#### 5.4.1 Full-trajectory chronology

`scripts/lag_chronology_analysis.py` computed, at full tick resolution for
all 6 runs, $e_{lag}=s_{ref}-s_{tip}$, $e_\perp$ (cross-track), $\|e\|$,
$\|u_{raw}\|$ (the pre-clip damped-least-squares command), absolute $q_1,q_2$,
and $h_{x_{max}}$; and, at a moderate stride across the full run (not just
two narrow windows), $E_N$ — the schedule-vs-state mismatch energy, §5.2's
corrected quantity, computed identically but at finer path coverage
(`tables/lag_chronology_per_tick.csv`, `tables/lag_chronology_EN_strided.csv`,
`figures/lag_chronology_combined.png`) **[hardware/model, direct]**. For
each quantity, its onset is the path progress at which it first departs,
and stays departed for 5+ consecutive samples, more than $3\sigma$ from its
own early-path ($s<20$mm) mean (`tables/lag_chronology_onset_summary.csv`):

| pairing | $e_{lag}$ | $\|e\|$ | $\|u_{raw}\|$ | $E_N$ | $q_2$ | $h_{x_{max}}<20$mm | exit |
|---|---|---|---|---|---|---|---|
| selective, contact | 61.6 | 61.1 | 62.0 | 58.0† | 62.6 | 63.0 | 63.2 |
| selective, no-contact | 48.1 | 48.1 | 51.9 | 58.8 | 69.9 | 73.4 | 73.5 |

(mean onset $s$ in mm across 3 reps; † only 1 of 3 contact reps crossed the
$E_N$ threshold inside the sampled window at all.)

**The proposed chronology holds, but asymmetrically between the two
pairings.** For the **no-contact** pairing, the order is clean and
well-separated — lag and error rise first ($\approx48$mm), then command
norm ($\approx52$mm), then schedule-vs-state mismatch energy ($\approx59$mm),
then $q_2$ drift ($\approx70$mm), then the margin crosses a 20mm threshold
and exits ($\approx73$-$75$mm) — a genuine $\approx25$mm-long staged
cascade, in exactly the proposed order. For the **contact** pairing, the
same events are compressed into a $\approx5$mm window immediately before
exit ($58$-$63$mm) — present, in the same relative order, but not
meaningfully *staged*; this pairing is a much weaker test of "does a
chronology precede the exit" than the no-contact pairing is. **A caveat on
the no-contact
pairing's own mean onset values: they mask real rep-to-rep spread** — one
of the three no-contact reps reaches these onsets $15$-$25$mm later than the
other two on several quantities (lag onset $64.6$mm vs $39$-$41$mm; $u_{raw}$
onset $67.9$mm vs $42$-$45$mm), and is also the one rep that exits earliest
($69.99$mm vs $75.32$mm for the other two, which reach the very end of the
reference before tripping the box) — the staged cascade is real but its
exact timing is not tightly reproducible rep to rep. Separately, absolute
$q_1$ (not the $\delta\chi$-vs-matched-MPC quantity §5.1 uses) departs from
its *own* early-path value much earlier ($32$-$39$mm, both pairings) than
$q_2$ does by this same-quantity test — this does not contradict §5.1's
finding (that $q_1$'s divergence *from MPC's own trajectory* comes late),
since MPC's own $q_1$ may be drifting similarly early; it is a different
reference, not a different fact.

#### 5.4.2 The most important comparison: does the wrong Jacobian cause more lag?

Directly comparing the 3 contact-Jacobian vs 3 no-contact-Jacobian selective
reps (full-run statistics, `tables/lag_chronology_onset_summary.csv`):

| quantity | contact $J_C$ | no-contact $J_{NC}$ | ratio |
|---|---|---|---|
| $e_{lag}$ mean (mm) | 0.39 | 1.09 | 2.8$\times$ |
| $e_{lag}$ median (mm) | 0.28 | 0.60 | 2.1$\times$ |
| $e_{lag}$ max (mm) | 3.05 | 5.57 | 1.8$\times$ |
| $e_\perp$ mean (mm) | 0.50 | 0.68 | 1.4$\times$ |
| $e_\perp$ max (mm) | 4.53 | 4.53 | 1.0$\times$ |

**This is neither of the two outcomes the chronology hypothesis
anticipated — it is a third, more precise one.** $J_{NC}$ does develop
substantially more **longitudinal lag** than $J_C$ (1.8-2.8$\times$ across
every statistic) — the wrong-Jacobian mismatch is a real, measurable
lag-amplifier. But **cross-track error's own maximum is essentially
identical between the two pairings** (4.53mm vs 4.53mm), and its mean is
only mildly higher for $J_{NC}$ (1.4$\times$, far below the lag ratio). Read
together with the shared-failure-face constraint above: the wrong Jacobian
measurably worsens *how far behind* the controller falls, but does not
explain *how large* the lateral/configuration error becomes at the point of
failure — that converges to nearly the same value regardless of which
Jacobian is used. This is consistent with treating the shared $x_{max}$-face
exit as architecture-intrinsic (present, and converging to the same
magnitude, under both Jacobian models, exactly as the "both pairings fail on
the same face" constraint above requires), with the wrong Jacobian acting as
a genuine but secondary amplifier of the *lag* component specifically, not
as an explanation of the common failure mode itself.

#### 5.4.3 Effective longitudinal gain: the mechanistic bridge

`scripts/effective_gain_and_baseline_analysis.py` computed, at each sampled
state, $M_k=J_C^{state}(x_k)\,\hat J_k^\dagger$ (the controller's own
schedule-based damped pseudo-inverse $\hat J_k^\dagger$, mapped through the
physically-grounded contact model) and projected it onto the local path
tangent $t_k$:
$$k_{\parallel,k}=K_p\,t_k^T J_C^{state}(x_k)\,\hat J_k^\dagger\,t_k,\qquad
k_{\perp,k}=\left\|(I-t_kt_k^T)J_C^{state}(x_k)\hat J_k^\dagger t_k\right\|$$
— the controller's *realized* closed-loop gain in the direction of travel,
as opposed to its *nominal* $K_p=0.6$
(`tables/k_parallel_k_perp_selective.csv`, `figures/k_parallel_vs_s.png`)
**[model, direct]**:

| pairing (s>45mm, post-contact) | $k_\parallel$ mean | $k_\parallel$ median | $n$ |
|---|---|---|---|
| selective, contact | 0.569 | 0.571 | 137 |
| selective, no-contact | 0.397 | 0.392 | 183 |

**The contact-Jacobian pairing's realized gain sits almost exactly at its
own nominal $K_p=0.6$ in the post-contact region; the no-contact pairing's
realized gain is $\approx30\%$ lower.** This is the mechanistic bridge the
chronology in §5.4.1-5.4.2 needed: $J_{NC}$ does not merely differ from
$J_C$ in the abstract — mapped through the controller's own pseudo-inverse
and evaluated against the real contacted response, it measurably reduces how
much of the commanded correction actually reaches the tip in the travel
direction, specifically in the region where the two models disagree.
$k_\perp$ does not differentiate the two pairings (mean $\approx0.21$-$0.22$
both, with a handful of large outliers in each) — the effect is specific to
the longitudinal direction, not a general gain inflation/deflation.

#### 5.4.4 Theoretical baseline and excess lag

**Which baseline formula is correct depends on exactly how this controller's
$K_p$ enters its control law, and was checked directly against source
rather than assumed.** `inverse_jacobian_controller.py`'s `solve()` computes
`task_velocity = pseudo @ (self.position_gain * (desired - measured) /
self.dt)` — i.e. the commanded velocity is
$\dot\chi=\hat J^\dagger(K_p\,e/\Delta t)$, so the position correction
actually realized over one control tick is
$\Delta p\approx J\dot\chi\,\Delta t=J\hat J^\dagger K_p\,e\approx K_p\,e$
(since $J\hat J^\dagger\approx I$ in task space for a well-conditioned,
full-row-rank $J$) — **the $\Delta t$ cancels out of the per-tick
correction, but the reference's own per-tick advance, $v_s\Delta t$, does
not**, since it is a genuinely discrete, one-tick quantity. The steady-state
balance between these two is $K_p\,e_{ss}=v_s\Delta t$, giving
$e_{ss}\approx v_s\Delta t/K_p$ — confirming, from the actual control law
rather than a generic assumption, that the $\Delta t$-inclusive baseline
below is the version applicable to this specific (discrete, no-feedforward)
implementation, not the continuous-time $v_s/K_p$ form that would apply to
a different control-law structure.

Using each run's own logged $K_p$ and the reference's own precomputed path
speed $v_s$ (no feedforward term exists in this controller: confirmed from
`solve()`, where the feedforward branches are both gated off in every run
analysed here), the baseline $e_{lag,expected}=v_s\cdot\Delta t/K_p$
gives an **excess lag** $e_{excess}=e_{lag,measured}-e_{lag,expected}$
(`tables/excess_lag_baseline.csv`) **[model, direct]**:

| region | contact $J_C$, mean excess (mm) | contact $J_C$, median | no-contact $J_{NC}$, mean excess (mm) | no-contact $J_{NC}$, median |
|---|---|---|---|---|
| pre-contact ($s\leq28.5$mm) | 0.086 | 0.071 | 0.035 | 0.064 |
| post-contact ($s>28.5$mm) | 0.180 | **0.023** | **1.273** | **0.557** |

**Pre-contact, both pairings sit close to the simple baseline and are
similar to each other** — consistent with H1's independent finding that the
two models agree before contact. **Post-contact, the contact pairing stays
close to baseline (its median excess is nearly zero — most ticks track at
essentially the baseline-predicted lag, with the mean pulled up by
occasional excursions), while the no-contact pairing's excess lag grows to
7$\times$ (mean) to 24$\times$ (median) the contact pairing's own value.**
This directly supports the reviewer's hypothesised mechanism: $J_{NC}$
develops lag well beyond what proportional gain alone explains, specifically
after contact onset — exactly where, and only where, the two physical models
disagree.

#### 5.4.5 Does raising the gain fix it? Revisiting the gain ablation through lag and cross-track

**This re-analysis is offered as generic evidence about the gain trade-off,
not as a direct test of the 255mm selective-gate failure itself — the
conditions differ on three axes at once** (no-contact Jacobian only vs both
pairings; hold gate vs selective gate; 210mm vs the 255mm floor where H3b's
failure occurs), and no run in this investigation varies gain at the actual
255mm/selective-gate condition. Whether raising $K_p$ would prevent or
worsen that specific failure was not tested and is not claimed here.

The gain ablation (§8, $K_p=0.6$ vs $1.0$, both no-contact Jacobian, hold
gate, 210mm) was originally used only to test whether a higher gain
increases infeasible-request pressure (it does not). Re-examined through
lag and cross-track specifically (`tables/gain_ablation_lag_reanalysis.csv`)
**[hardware, direct]**:

| gain | lag mean (mm) | lag median (mm) | lag p95 (mm) | cross-track mean (mm) | cross-track p95 (mm) |
|---|---|---|---|---|---|
| 0.6 | 0.395 | 0.360 | **0.979** | 0.210 | 0.563 |
| 1.0 | 0.300 | 0.271 | **1.398** | 0.404 | 1.429 |

**Raising $K_p$ reduces *typical* lag (mean $-24\%$, median $-25\%$) but
makes the lag distribution's own *tail* worse (p95 $+43\%$), while
cross-track degrades substantially at both the mean ($+92\%$) and especially
the tail ($+154\%$).** This matches the reviewer's first anticipated
outcome — gain trades lag for oscillation — with a sharper edge than
anticipated: it is not simply "typical lag for oscillation," it is "modest
typical-lag improvement, at the cost of a worse lag tail *and* substantially
worse lateral tracking." **Stated to match what this specific data supports
and no more: existing gain-ablation data (at 210mm, hold gate, no-contact
Jacobian) show that increasing proportional gain reduces typical lag but
substantially worsens tail and lateral tracking; therefore higher gain is
not a cost-free remedy at the condition actually tested. Whether it would
prevent the specific 255mm selective-gate workspace failure this subsection
is otherwise concerned with was not tested.** (A fourth statistic,
final-tick lag, is not reported here: inspection of the per-tick series
shows the controller settles into a `terminal_hold` state once the
reference completes, where $e_{lag}$ repeatedly returns to a near-identical
small residual — $0.0628$mm, reproduced to 6 decimal places across several
ticks and across different runs — reflecting the polyline projection
geometry of the terminal reference point rather than any property of the
gain being tested; using it as a summary statistic would be misleading.)

#### 5.4.6 Margin-rate decomposition and a same-state $K_p$ counterfactual, at the actual failure condition

§5.4.5's gain ablation is evidence about the generic lag-vs-oscillation
trade-off, but not a test *at the condition that actually fails* (255mm,
selective gate) — stated explicitly there as an open question. This
subsection closes that gap directly, for the contact-Jacobian selective
runs only, by replacing correlational chronology with an instantaneous,
per-tick causal decomposition and a same-state $K_p$ counterfactual at the
real failure condition itself.

**The decomposition.** At every tick, the realized one-tick change in the
binding $x_{max}$ margin is predicted from the realized configuration
change via the margin's own gradient,
$\Delta h_{x_{max},k}\approx\nabla_\chi h_{x_{max}}^T\Delta\chi_{actual,k}$
(finite-difference gradient of the flange forward kinematics — cheap, no
live contact-model evaluation needed), decomposed per joint,
$\Delta h_i=\partial h_{x_{max}}/\partial\chi_i\cdot\Delta\chi_{actual,i}$
(`scripts/h3b_margin_rate_decomposition.py`,
`tables/h3b_margin_rate_decomposition.csv`,
`figures/h3b_margin_rate_chronology.png`) **[hardware + model, direct]**.
**This linearized prediction is verified against the real observed margin
change before being used as evidence**: correlation between predicted and
actual one-tick $\Delta h_{x_{max}}$ across all 1268 sampled ticks (3 reps)
is $0.911$, mean absolute error $0.43$mm — the gradient decomposition is a
good local model of what actually happened, not an unchecked assumption.

Plotting $e_{lag}$, $\|u_{raw}\|$, $\|\Delta\chi_{actual}\|$,
$\Delta h_{x_{max}}$, and $h_{x_{max}}$ itself together shows a richer
picture than a single clean three-stage cascade: $e_{lag}$, $\|u_{raw}\|$,
and $\|\Delta\chi_{actual}\|$ track each other almost exactly throughout (as
they must, being closely related quantities) and rise sharply only in the
last $\approx5$mm before exit ($s\approx58$-$64$mm) — consistent with
§5.4.1's finding that this chronology is compressed, not staged, for the
contact pairing. $h_{x_{max}}$ itself, however, declines **almost
monotonically from the start of the path** ($\approx330$mm at $s\approx8$mm
down to near zero at exit), and $\Delta h_{x_{max}}$ is mildly but
persistently negative through most of the middle of the path
($s\approx15$-$55$mm, averaging roughly $-0.5$ to $-1$mm/tick) well before
the late command surge — i.e. there is a slow, steady baseline erosion of
this margin for most of the path, **on top of which** the late lag/command
surge adds a much sharper, accelerating negative component
($-5$ to $-7$mm/tick at $s>58$mm) that produces the actual exit. Both
components are real and both are shown directly; neither is claimed to be
the sole explanation.

**The per-DOF decomposition directly confirms the causal claim, not just
correlates with it.** In the late window ($s>55$mm), joint $q_1$'s own mean
contribution to $\Delta h_{x_{max}}$ is $-1.57$mm/tick — consistently
negative (mean $\approx$ mean absolute value, i.e. essentially never
positive in this window) and $5$-$9\times$ larger in magnitude than any
other joint's contribution ($q_2$: $-0.18$, $q_3$: $-0.29$, $q_4$: $-0.10$,
$q_5$: $-0.10$, $q_6$/insertion: exactly $0$, since the $x_{max}$ margin does
not depend on insertion). **This directly establishes, rather than merely
correlates with, the claim that the growing commands increasingly move
$q_1$ specifically in the direction that reduces the $x_{max}$ margin** —
consistent with, and now causally grounding, §5.1's independent finding
that $q_1$ dominates this margin's linear sensitivity.

**The $K_p$ counterfactual.** At the same late-window states
($s>45$mm, $n=410$ ticks across 3 reps), the controller's raw command was
recomputed with $K_p=1.0$ in place of its own logged $K_p=0.6$, holding the
measured state, the reference, and the real logged recursive
previous-command history fixed (the same "freeze everything else, vary one
input" convention as every other same-state counterfactual in this report).
Because $\text{task\_velocity}=\hat J^\dagger(K_p\,e/\Delta t)$ and neither
$\hat J^\dagger$ nor $e$ depends on $K_p$, the pre-clip command scales
exactly linearly in $K_p$; the same velocity/acceleration/state-box clip is
then reapplied before evaluating the margin change
(`tables/h3b_kp_counterfactual_margin_rate.csv`,
`figures/h3b_kp_counterfactual_margin_rate.png`) **[counterfactual]**:

| quantity | $K_p=0.6$ (actual) | $K_p=1.0$ (counterfactual) |
|---|---|---|
| mean $\Delta h_{x_{max}}$ (mm/tick) | $-1.32$ | $-1.97$ |
| mean $\|\Delta h_{x_{max}}\|$ (mm/tick) | $1.32$ | $1.97$ |
| fraction of ticks where $K_p=1.0$ is more negative | — | $0.988$ |

**At 98.8% of the 410 sampled ticks, the higher-gain counterfactual command
produces a *more negative* margin step than the actual $K_p=0.6$ command did
at the identical state** — a $\approx49\%$ larger mean margin loss per tick.
This is the decisive result the chronology and the generic gain ablation
could only gesture at: **at the actual failure condition, same states,
same history, changing only $K_p$, higher gain does not merely fail to
help — it makes each tick's workspace-margin consumption measurably
worse.** Combined with §5.4.4's finding that higher gain reduces lag (by
construction — larger $K_p$ closes the tracking-error-proportional command
faster) and §5.4.5's finding that it worsens lag's own tail and cross-track
at a different condition, this closes the loop precisely as hypothesised:
**low gain produces more following lag, but higher gain produces more
aggressive, workspace-margin-consuming corrections — which is why simply
raising $K_p$ is not a solution to this failure mode, not merely an
unexplored option.**

#### 5.4.7 Synthesis

$$
\boxed{
\begin{aligned}
&K_p=0.6\text{, moving reference}\ \rightarrow\ \text{baseline tracking lag (both pairings, pre-contact; }\\
&\text{contact pairing stays near this baseline even post-contact, §5.4.4)}\\
&\downarrow\ \text{(no-contact pairing only, from here)}\\
&J_{NC}\text{ reduces realized longitudinal gain }\approx30\%\text{ specifically post-contact (§5.4.3)}\\
&\rightarrow\ \text{excess lag }7\text{-}24\times\text{ the contact pairing's, same region (§5.4.4)}\\
&\downarrow\\
&e_{lag}\uparrow\ (\approx48\text{mm})\ \rightarrow\ \|u_{raw}\|\uparrow\ (\approx52\text{mm})\ \rightarrow\ E_N\uparrow\ (\approx59\text{mm})\\
&\rightarrow\ q_2\text{ drift }(\approx70\text{mm})\ \rightarrow\ h_{x_{max}}\downarrow\ \rightarrow\ \texttt{tcp\_out\_of\_workspace}\ (\approx74\text{mm})\\
&\text{(this downstream chain: well-staged for no-contact, §5.4.1; compressed}\\
&\text{into a 5mm pre-exit window, same order, for contact; }E_N\text{ here is the}\\
&\text{schedule-vs-state mismatch energy of §5.2, not control-law null-space motion)}
\end{aligned}
}
$$

**What this does and does not establish.** It establishes a measured,
mechanistic account of *why* the no-contact pairing develops more lag than
the contact pairing (reduced realized longitudinal gain, §5.4.3), and that
this additional lag precedes, in a well-separated staged order, the same
downstream command-growth/mismatch-growth/configuration-drift/exit sequence
§5.1 (configuration drift) and §5.2 (schedule-vs-state mismatch, corrected)
already established for the endpoint. **It does not establish that this
mechanism is the explanation for the shared failure mode**, precisely
because the contact pairing — whose realized gain stays near nominal and
whose excess lag stays near baseline throughout — still exits on the
identical $x_{max}$ face, with the same terminal cross-track magnitude, on a
compressed version of the same event order. The honest reading: the wrong
Jacobian is a real, quantified, mechanistically-grounded *lag amplifier*,
operating on top of an architecture-intrinsic configuration-drift mechanism
(§5.1, the direct $q_1/q_2$-vs-matched-MPC measurement, present under both
Jacobian models) that remains the primary candidate explanation for the
shared failure itself — and that mechanism's own evidentiary basis is now
§5.1's joint-angle measurement alone, not the $E_N$ analysis, per §5.2's
correction.

**§5.4.6 closes one specific link in this chain causally rather than only
architecturally**: the step "larger commands move $q_1$ in the margin-
reducing direction" is no longer only inferred from §5.1's correlational
sensitivity ranking — it is directly measured, tick by tick, via a
gradient decomposition verified against the real observed margin change
($r=0.911$), and $q_1$'s own contribution dominates every other joint's by
$5$-$9\times$ in the failure window. The companion $K_p$ counterfactual
closes the gain question specifically: at the real failure condition, same
states, same history, only $K_p$ changed, higher gain makes the margin
worse at 98.8% of sampled ticks — so "low gain causes lag, high gain causes
more aggressive margin-consuming corrections" is now a directly demonstrated
trade-off at the condition that actually fails, not an inference carried
over from a different (210mm, hold-gate) condition.

---

## 6. H4 — Scheduled Jacobian accuracy against the contact model at the measured state

Sections 6.1-6.4 of `STAGE3_FINAL_REPORT.md` (why $J_C^{state}$ is the only
physically-grounded reference; the whole-run Q1/Q2/Q3 tables: relative-
Frobenius mismatch, command-weighted error, gain ratio, direction angle; and
the staleness-vs-missing-physics secondary decomposition, there numbered
§6.5) are **unchanged from the prior report** and are not repeated here. The
headline whole-run finding stands: the no-contact-scheduled MPC condition at
the tight floor is the outlier on every one of $E_J$, $\epsilon_u$, and the
2.64$\times$ gain ratio simultaneously, and its mismatch-vs-tracking-error
correlation moves from near-zero at the loose floor to +0.47 at the tight
floor.

### 6.5 New: common-progress-matched bins, and temporal ordering against H2's failure

Whole-run means above are vulnerable to one specific distortion: the
255mm MPC-$J_{NC}$ condition terminates at $s\approx64$mm, while its
contact-Jacobian comparator completes the full $75.3$mm reference path (note:
$75.3$mm is path progress $s$, the arc-length coordinate used throughout this
report — not to be confused with insertion length $L$, which this same
255mm contact-aware condition reaches $89.7$-$91.0$mm of, §3.4) — a
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
temporal-ordering evidence relevant to §3.4's causal chain: the model
mismatch is not something that only appears alongside the terminal tracking
blow-up, it is already present and already larger than the matched
comparator's own mismatch tens of millimetres of path progress earlier.
This is **consistent with, but does not establish**, the §3.4 causal chain —
the same-state counterfactual attempted in §3.3 for the directional question
is inconclusive (see there); this bin-matched table remains a magnitude-only,
temporal-ordering observation, not a demonstration of causal direction.

### 6.6 Robustness of the gain-ratio statistic

A mean of a ratio can be distorted by a small number of near-zero-
denominator ticks, so the gain-ratio figures above — especially the 60-64mm
bin's 7.94$\times$, used repeatedly elsewhere in this report — were checked
against two more robust statistics computed on the same per-sample data,
re-deriving the numerator and denominator norms separately rather than
reading only the stored ratio: the **median** (insensitive to outliers in
either direction) and $G_{agg}=\sum_k\|J_{sched}\Delta\chi_k\|/\sum_k\|J_C^{state}\Delta\chi_k\|$
(an aggregate, motion-weighted ratio, insensitive to a few near-zero-
denominator ticks inflating an individual ratio — though, as this check
found, itself vulnerable to the opposite failure mode: a single
*anomalously large* denominator can dominate its sum and collapse the
statistic) (`scripts/h4_gain_ratio_robustness.py`,
`tables/h4_gain_ratio_robustness.csv`, `figures/h4_gain_ratio_robustness.png`)
**[model, re-derived]**:

| bin (mm) | condition | n | mean (original) | median | $G_{agg}$ |
|---|---|---|---|---|---|
| pre-contact (0-28.5) | MPC-$J_C$ | 37 | 1.28 | 1.22 | 1.34 |
| pre-contact (0-28.5) | MPC-$J_{NC}$ | 45 | 1.29 | 1.17 | 1.26 |
| early contact (28.5-45) | MPC-$J_C$ | 82 | 1.48 | 1.44 | 1.46 |
| early contact (28.5-45) | MPC-$J_{NC}$ | 107 | 2.44 | 2.44 | 2.42 |
| mid-contact (45-55) | MPC-$J_C$ | 61 | 1.59 | 1.36 | 1.90 |
| mid-contact (45-55) | MPC-$J_{NC}$ | 60 | 2.15 | 2.11 | **0.07** |
| late common (55-60) | MPC-$J_C$ | 29 | 1.42 | 1.40 | 1.54 |
| late common (55-60) | MPC-$J_{NC}$ | 27 | 2.41 | 2.24 | 2.28 |
| NC pre-failure (60-64) | MPC-$J_C$ | 12 | 1.91 | 1.82 | 1.94 |
| NC pre-failure (60-64) | MPC-$J_{NC}$ | 19 | **7.94** | **7.30** | **7.19** |

**The headline figure survives**: in the 60-64mm bin, mean (7.94), median
(7.30), and $G_{agg}$ (7.19) all agree closely — this is a robust, large
effect, not an artefact of a few extreme ticks. The 28.5-45mm and 55-60mm
bins are similarly consistent across all three statistics (within
$\approx10\%$ of each other).

**The mid-contact (45-55mm) bin's $G_{agg}=0.07$ is not a real effect — it
is a single-tick numerical artefact, traced and confirmed, not left as an
unexplained anomaly.** Inspecting the 60 underlying per-sample rows
directly (`tables/h4_gain_ratio_robustness_per_sample.csv`) identifies
exactly one tick ($s=47.91$mm) whose live-model denominator,
$\|J_C^{state}\Delta\chi\|=423.7$mm, is physically impossible for a single
0.1s control tick in a system whose typical one-tick tip motion is
$0.1$-$0.3$mm — three orders of magnitude larger than every other sample in
the entire 490-row 255mm dataset (next-highest: $0.28$mm). This is a live
Jacobian evaluation anomaly at that one state (plausibly a near-singular or
otherwise pathological configuration passed to the contact-model solve), not
a real physical response, and because $G_{agg}$ sums denominators before
dividing, this single outlier dominates and collapses the bin's aggregate
ratio. **Excluding only this one tick, the bin's three statistics converge
to mutual agreement**: mean $2.13$, median $2.13$, $G_{agg}=2.12$ — fully
consistent with the mean/median already reported, and with the neighbouring
bins' elevated-but-not-yet-extreme readings. The §6.5 temporal-ordering
claim (mismatch already elevated in the 45-55mm and 55-60mm bins, before the
60-64mm failure bin) is therefore robust across all three bins once this one
diagnosed artefact is excluded.

---

## 7. H5 — Scheduled vs frozen Jacobian

Sections 7.1 (design: why two frozen ablations, chosen to separate
conditioning from staleness) and 7.2 (headline completion/RMSE table) are
**unchanged from the prior report**. The verdict stands: neither tested fixed
linearization matched the scheduled Jacobian's combination of tracking
quality and robust completion.

### 7.3 Why index 0 survives and index 100 does not — a same-state counterfactual was attempted, and is inconclusive

The prior report's directional-alignment check (index 0's dominant output
direction staying closer to the schedule's own evolving direction than
index 100's, at every post-contact checkpoint) is correlational, and remains
the only evidence this report can currently stand behind for *why* index 0
survives and index 100 does not.

**A same-state counterfactual was built and run, using the same 11 states
from the frozen-idx100 run's own hardware log as before, and is reported
here, but its original "useful correction" metric did not survive a sign
check and the finding is revised to inconclusive as a result — this is a
revision made during this pass, not a result carried over from an earlier
version of this report.** The counterfactual itself (re-solving the identical
MPC at each state with the real frozen-idx100 schedule, the trajectory-
varying schedule's own matrix at that reference index, and the frozen-idx0
matrix, all three commands evaluated through $J_C^{state}(x_k)$) is sound —
sanity-check replay error against the logged command is under 0.05% at every
state. What was wrong was the metric applied to the three resulting
commands.

**The problem.** The original metric was
$c_\parallel=\hat e_k^T J_C^{state}\Delta\chi$ with
$\hat e_k=(p_{ref,k}-p_{tip,k})/\|p_{ref,k}-p_{tip,k}\|$, read as "useful
correction," with its sign explicitly discarded ("a negative/positive
alternation tick to tick is expected and not itself meaningful; magnitude is
the relevant quantity"). That is not a valid reading: $c_\parallel>0$ means
predicted motion *toward* the reference and $c_\parallel<0$ means predicted
motion *away* from it, under the stated definition — the sign is exactly the
information that distinguishes a genuinely useful command from a harmful
one, and discarding it before reporting "a 50-200$\times$ gap" was an error
that this report now corrects rather than repeats.

**The attempted fix, and why it is reported as inconclusive rather than
successful.** Replacing $c_\parallel$ with a signed one-step predicted error
reduction, $\Delta e=\|e_{before}\|-\|e_{after}\|$ (using the *next*
reference sample $p_{ref,k+1}$ in both $e_{before}$ and $e_{after}$, still
evaluated through $J_C^{state}$), gives a metric whose sign is unambiguous —
but it reverses the qualitative story rather than confirming it:

| $s$ (mm) | idx100 actual, OLD $c_\parallel$ | idx100 actual, NEW $\Delta e$ | sched c/f, OLD $c_\parallel$ | sched c/f, NEW $\Delta e$ | idx0 c/f, OLD $c_\parallel$ | idx0 c/f, NEW $\Delta e$ |
|---|---|---|---|---|---|---|
| 20 | −0.085 | −0.022 | −0.179 | −0.126 | −0.031 | −0.079 |
| 30 | 0.192 | 0.207 | 0.604 | 0.152 | 0.115 | 0.034 |
| 40 | 0.105 | 0.119 | **0.866** | **−0.428** | 0.083 | 0.056 |
| 50 | −0.128 | −0.038 | −1.064 | −1.707 | −0.211 | −0.126 |
| 55 | 0.110 | 0.109 | **1.826** | **−1.904** | 0.166 | 0.158 |
| 58 | −0.101 | −0.110 | −1.396 | −1.653 | −0.184 | −0.183 |
| 60 | 0.108 | 0.097 | **1.118** | **−0.097** | 0.137 | 0.139 |
| 62 | −0.168 | −0.153 | −1.665 | −2.029 | −0.237 | −0.224 |
| **64** | **0.016** | **0.079** | **3.133** | **−3.096** | **0.063** | **0.058** |
| 67 | −0.084 | −0.086 | **0.820** | **−2.765** | −0.075 | −0.107 |
| 70 | 0.062 | 0.063 | 0.204 | −0.281 | 0.140 | 0.141 |

(`tables/h5_signed_error_reduction.csv`, `figures/h5_signed_error_reduction.png`
— computed alongside the original table, `tables/h5_same_state_counterfactual.csv`,
`figures/h5_same_state_frozen_counterfactual.png`.) **[counterfactual]**

Under this corrected, sign-respecting metric:

1. **The scheduled counterfactual's $\Delta e$ is negative — a predicted
   *worsening*, not an improvement — at 9 of 11 states**, including the
   exact $s=64$mm state previously used as the headline illustration: it
   flips from $+3.13$mm (read, incorrectly, as "far more correction") to
   $-3.10$mm (a comparably large predicted worsening) under the corrected
   metric. **The "scheduled dominates" claim, as quantified by either
   version of this metric, does not survive** and is withdrawn.
2. **idx0 no longer clearly beats idx100's own actual command in the
   late-path region.** Of the 8 marks at $s\geq50$mm, idx100's own command
   now has a larger (less negative, or more positive) $\Delta e$ than idx0's
   counterfactual at 5 of 8 — including $s=64$mm, where idx100
   ($+0.079$mm) now exceeds idx0 ($+0.058$mm), the reverse of the original
   illustration. idx0 is clearly ahead only at $s=55,60,70$mm. **The
   staleness mechanism this same-state counterfactual was built to
   demonstrate directly is not established by it.**
3. All three commands remain small and comparable in magnitude throughout
   (roughly 0.02-0.22mm for idx100 and idx0; the scheduled command is
   consistently larger in magnitude, 0.1-3.1mm, but — per point 1 — its sign
   is as often a predicted worsening as an improvement).

**Why this happened, and why it is reported as inconclusive rather than as a
negative result for H5's mechanism specifically.** The scheduled command is
the only one of the three built from a model meant for live, delay-
compensated redeployment — the real MPC re-linearizes every horizon step and
is delay-aware (it carries a `delay_samples` parameter and solves over a
multi-step horizon). idx100-actual and idx0-counterfactual are both frozen,
single-matrix commands with no comparable delay compensation at this
instant. The scheduled command's much larger magnitude, and its tendency to
flip sign under a *one-step-ahead* metric, is consistent with it being
intended to correct a predicted *future* tracking state rather than the
immediate next tick — exactly the delay/horizon mismatch an external review
of this report flagged as a concern with projecting onto the *current*
tracking-error direction at all. The $\Delta e$ metric above is itself only
a one-step-ahead approximation (it still does not propagate the solved
trajectory through the controller's own horizon/delay buffer to compute the
actual change in predicted multi-step tracking cost, $\Delta V_{track}$,
which would be the fully rigorous fix) — so neither the original nor the
corrected metric can be trusted to isolate the mechanism for a command whose
intended effect is multiple steps ahead.

**Net status of this subsection: the outcome-level facts from §7.2 (scheduled
completes; idx0 completes but tracks worse; idx100 tracks well early, then
fails late) are direct hardware observations and are unaffected by any of
this.** The *mechanistic* claim — *why* index 0 survives and index 100 does
not — is supported only by the correlational directional-alignment check
carried over from the prior report (§7.2); the same-state counterfactual
mechanism attempted in this subsection is reported as **attempted and
inconclusive**, not as a demonstrated mechanism, and a genuine resolution
would need the full delay/horizon-aware $\Delta V_{track}$ metric, which
remains an open item (§11).

---

## 8. Secondary ablation — proportional gain (unchanged)

Unchanged from the prior report's §8. Verdict: contradicted — the
infeasible-request rate is exactly zero at both gains tested; the
tracking-degradation pattern at the higher gain (larger, more oscillatory
commands, 34% more sign reversals) is consistent with, but not proven to be,
underdamped behaviour, and is independent of the exclusion constraint
entirely.

---

## 9. The $s\approx57$-$61$mm local transient in scheduled contact-aware MPC: expanded diagnostic

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
  reps**, so a contact-state check at the precise peak tick itself remained
  a residual gap at this point in the investigation — closed below.

### 9.1 Closing the residual gap: exact-peak-tick contact state, and a same-state schedule-vs-state replay

Two further, exact (not coarse-stride) checks were run at the precise
`is_peak==True` tick identified per rep (`tables/transient_peak_refine.csv`),
closing out the two items flagged as incomplete above.

**Exact-tick contact state.** Recomputing the contact model's gap and
contact-force quantities at the exact peak tick in all three reps (not its
coarse-stride neighbours) gives gap $35$-$37\mu$m and force $1.56$-$1.70$N —
smooth, consistent with the immediately surrounding ticks, no penetrating
nodes. **This adds no new information**: the exact-tick values confirm,
rather than contradict, the coarser-stride finding already reported above.

**Exact-tick same-state MPC replay — this is new, and changes the
conclusion.** At the same three exact peak ticks, the real tick was first
re-solved with no override (sanity-checking the replay against the logged
command: relative error $0.011$-$0.017\%$ in reps 2 and 3, but $20.65\%$ in
rep 4 — a genuine solver-sensitivity outlier at that specific tick, which
needed $1350$ QP iterations against $500$-$1800$ for neighbouring ticks, and
is reported with this caveat rather than treated as equally reliable), then
re-solved again with the scheduled Jacobian $J_C^{sched}$ replaced by
$J_C^{state}(x_k)$, the contact model recomputed at that exact measured
state (`tables/transient_exact_peak_closure.csv`) **[counterfactual]**:

| rep | $s$ (mm) | sanity rel. err. | $\|u_0^{sched}\|$ | $\|u_0^{state}\|$ | relative command difference |
|---|---|---|---|---|---|
| 2 | 60.24 | 0.017% | 0.068 | 0.210 | **211%** |
| 3 | 60.47 | 0.011% | 0.060 | 0.217 | **262%** |
| 4 | 60.24 | 20.65% (flagged, lower confidence) | 0.079 | 0.235 | 199% |

**Swapping the scheduled Jacobian for the state-recomputed one at the exact
peak tick changes the command by roughly $2$-$2.6\times$ in norm** — far
larger than the schedule-vs-state mismatch found anywhere else in this
investigation, and present in all three reps (though rep 4's number carries
the sanity-check caveat above). Measuring the matrix-level mismatch directly
at these same three ticks, $E_J=\|J_{sched}-J_C^{state}\|_F/\|J_C^{state}\|_F$,
gives $0.65$-$0.66$ in all three reps — roughly double this condition's own
whole-run mean ($0.316$ at $210$mm, H4 §6.2).

**This does not contradict the original eight-candidate check** (which
included a schedule-vs-live-state divergence quantity, found unremarkable
over $s=54$-$60$mm) — it resolves the same issue the peak-location
correction above already flagged: that check, like the original tracking-
error search, was run over a window that ends at $s=60$mm and never actually
reached the true peak at $s\approx60.2$-$60.5$mm. Checked at the right tick,
the previously-dismissed candidate turns out to be anomalous after all.

**Conclusion.** Of the twelve previously-checked candidates plus the two
closed here, **a candidate mechanism is now identified**: an anomalously
large schedule-vs-state-recomputed Jacobian mismatch, specific to the exact
peak tick, that materially changes the commanded correction when tested by
actual re-solved command (not only by matrix norm). This is supported at the
matrix level in all 3 reps and at the command-replay level in 2 of 3 reps
with high-confidence replay fidelity (rep 4's replay itself has a large
baseline discrepancy at this exact tick and is reported with that caveat,
not used to anchor the claim). Vision/measurement (1A), reference geometry
(1B), and execution lag (1C) remain ruled out. **This is reported as a
repeatable candidate mechanism, not yet an exhaustively characterized one**:
a planned finer sweep of $E_J$ across the neighbouring ticks (to determine
whether this is a sharp, isolated spike or a broader plateau) did not finish
in time and remains open (§11). The transient remains small in absolute
terms (under 3.81mm, below every abort threshold used in this report) and
does not threaten any conclusion elsewhere in this document, but it is no
longer accurately described as "unexplained."

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
tens of millimetres before the eventual tracking failure (H4, §6.5), and
the command it actually issues at matched states differs substantially in
magnitude from what a contact-informed model would command there (H2,
§3.3) — with every workspace and exclusion margin confirmed to stay
generous throughout, ruling out a configuration-space-exhaustion reading
(§3.2). **Whether that command difference makes the realized tracking
better or worse is not established**: a same-state counterfactual was built
for exactly this question, using a one-step signed error-reduction metric,
and found the result sensitive to a known limitation of that metric (it
cannot account for MPC's delay-aware, multi-step horizon) rather than
settling the question — reported as attempted and inconclusive, not as a
demonstrated mechanism (§3.3). The same limitation applies to H5's analogous
counterfactual: the claim that the scheduled Jacobian's command realizes
more correction than either frozen alternative, and that idx0 outperforms
idx100 late in the path, does **not** survive the same sign correction (§7.3)
and is likewise withdrawn as a directly-demonstrated mechanism. What
survives for H5 is the outcome-level finding only: no single fixed
linearization (of the two tested) matched the scheduled Jacobian's
combination of tracking quality and robust completion, and the
*correlational* directional-alignment check from the prior report (index 0's
output direction staying closer to the schedule's own evolving direction)
remains the only evidence offered for *why*.

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
remains of it (§5.1) — the direct, Jacobian-independent measurement
underlying "accumulated redundant-configuration drift," the precise,
evidence-matched term, not "null-space drift." A companion quantity, $E_N$,
grows 10-17$\times$ from early path progress to the pre-failure window
(never exceeding $\approx40\%$ of the command's squared norm); this was
originally reported as a measurement of the control law's own null-space
behaviour, but a direct check (§5.2) found that premise mathematically
impossible for a damped-least-squares command projected against its own
Jacobian, and traced $E_N$'s actual meaning to schedule-vs-state Jacobian
mismatch channelled through the command — still a real, growing signal,
corroborating §5.1 temporally, but not itself evidence of null-space
behaviour in the control law. MPC's in-QP
constraint is directly observed to keep every tick feasible (§4.1, never
violated on any MPC tick analysed) — that part is a direct measurement. The
further claim that its input and input-increment costs are *why* its
configuration stays bounded while the inverse-Jacobian controller's drifts is
not separately isolated by an ablation in this investigation (no run varies
those cost weights while holding everything else fixed): it is **consistent
with** the observed contrast — MPC stays bounded, the naive controller does
not, and MPC's cost structure penalizes every commanded direction while the
naive controller's redundancy resolution at zero gain has no analogous term
— but is reported as an interpretation consistent with the evidence, not as
a directly demonstrated causal mechanism. A full-trajectory chronology
(§5.4) now shows this drift is preceded, in a staged and well-separated
order for the no-contact pairing (lag $\to$ command growth $\to$
mismatch-energy growth $\to$ configuration drift $\to$ exit), by a
measurable reduction in
the controller's *realized* longitudinal closed-loop gain where the
Jacobian is wrong ($k_\parallel$ down $\approx30\%$ post-contact, §5.4.3) —
but the contact-Jacobian pairing, whose realized gain stays near nominal
throughout, still exits on the same face via a compressed version of the
same sequence, so the wrong Jacobian is established as a quantified
*lag amplifier* on top of this architectural drift, not as its cause.

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
  relative error, but is not universally exact**: 1 of 20 original
  validation ticks showed a 123% relative error on a near-zero-magnitude
  command, attributable to a weakly-penalized QP direction rather than a
  replay defect; separately, 1 of the 3 exact-peak-tick replays in §9.1
  showed a 20.65% sanity error, plausibly a warm-start-sensitive hard QP
  instance at that specific tick (1350 solver iterations against 500-1800
  elsewhere). Every same-state result in this report reports its own
  per-tick sanity-check error alongside it, and conclusions are weighted
  toward ticks with low replay error where the two disagree (e.g. §9.1's
  rep 4).
- **The same-state counterfactual metric used for H2 (§3.3) and H5 (§7.3)
  is not resolved, and this is the most significant open methodological
  issue in this report.** Both subsections originally scored candidate
  commands with $c_\parallel=\hat e_k^T J\Delta\chi$ and explicitly discarded
  its sign; correcting that with a signed one-step error-reduction metric
  ($\Delta e$, using the next reference sample) reverses rather than
  confirms both subsections' original headline claims. Because MPC is
  delay-aware and solves over a multi-step horizon, a one-step metric of
  either kind is not a trustworthy judge of a command intended to correct a
  *future*, not current, tracking state — so the corrected metric's
  reversal is reported as evidence the one-step approach itself is
  inadequate, not as a new positive finding in the opposite direction. A
  genuine resolution requires propagating each candidate command through
  the controller's own multi-step horizon and delay buffer to compute the
  actual change in predicted tracking cost, $\Delta V_{track}$ — not
  attempted in this investigation. Until that is done, **H2's and H5's
  same-state mechanism claims should both be read as attempted and
  inconclusive**, not as directly demonstrated, despite the same-state
  replay infrastructure itself (command reproduction, magnitude
  comparisons) being independently validated and trustworthy.
- **Abort thresholds differ by controller architecture** (5mm MPC / 20mm
  inverse-Jacobian / 10mm open-loop) — unchanged from the prior report.
- **The $s\approx60$-$61$mm local transient has moved from unexplained to a
  candidate-mechanism-identified status (§9.1), but is not yet exhaustively
  characterized.** A planned finer sweep of the schedule-vs-state mismatch
  ($E_J$) across ticks neighbouring the exact peak, to determine whether the
  anomaly is a sharp spike or a broader plateau, did not complete in time.
  The mechanism itself (schedule-vs-state Jacobian mismatch) is supported
  at the matrix level in all 3 reps and at the command-replay level in 2 of
  3 (rep 4's replay has the sanity-check caveat above).
- **Single robot, single vessel geometry, single beam** (unchanged).
- **H4's gain-ratio figures were checked for denominator robustness (§6.6)
  and the headline 60-64mm figure survives**; one of the three "already
  elevated before failure" bins (45-55mm) showed an apparent inconsistency
  that traced to a single anomalous live-model evaluation at one tick
  ($s=47.91$mm, a denominator three orders of magnitude outside the rest of
  the 490-sample dataset) — diagnosed, excluded, and confirmed not to
  change the bin's reading once removed. This was resolved during this
  pass, not left open.
- **Would any of the above benefit from new hardware?** Still no — the
  remaining open items (a full $\Delta V_{track}$ implementation for H2/H5,
  a finer $E_J$ sweep for the transient) are answerable from existing logs
  and further offline/same-state computation. None requires a new
  closed-loop trial.

---

## 12. Final conclusions

| Hypothesis | Verdict | Strongest evidence | Mechanism status |
|---|---|---|---|
| **H1** — contact-aware planning required for an executable open-loop plan | **Strongly supported** | 2/2 contact-aware reps complete vs 0/2 no-contact reps (4 runs total); divergence lags predicted contact onset by $\approx22$mm | Directly demonstrated (model-vs-model comparison at matched states) |
| **H2** — contact-Jacobian × exclusion-radius interaction | **Supported** | completion asymmetry (3/3$\to$3/3 vs 3/3$\to$0/3); workspace/exclusion margins confirmed generous throughout (ruling out a workspace-exhaustion reading); H4's gain-ratio mismatch elevated before the terminal failure; same-state replay shows the actual command differs substantially in magnitude from a contact-informed one at every examined state | Magnitude-level facts (completion, margins, command-difference size) directly demonstrated; the *directional* claim ("the wrong model's command is less effective") was attempted via same-state counterfactual and found inconclusive (§3.3) — not directly demonstrated |
| **H3a** — MPC vs inverse-Jacobian intrinsic constraint awareness | **Strongly supported** | exact (1.000 agreement) raw-command recovery; MPC never violates its own in-QP constraint on any tick | Directly demonstrated |
| **H3b** — selective-gate robustness test + redundant-configuration drift | **Supported as a second, independent failure mode** | 0/6 complete under selective gating despite near-zero exclusion-gate activity; exact binding face ($x_{max}$) and two-stage per-DOF mechanism identified via direct joint-angle measurement (§5.1); a companion schedule-vs-state-mismatch quantity ($E_N$) grows 10-17$\times$ pre-failure, correlating temporally (§5.4); a full lag→command→mismatch→drift→exit chronology is staged and well-separated for the no-contact pairing, compressed but same-ordered for contact (§5.4); a margin-rate gradient decomposition (verified against ground truth, $r=0.911$) shows $q_1$ dominates the margin-reducing command direction by $5$-$9\times$ every other joint (§5.4.6); a same-state $K_p=0.6$-vs-$1.0$ counterfactual at the real failure condition shows higher gain makes the margin step worse at 98.8% of sampled ticks (§5.4.6) | Directly demonstrated via workspace-face/DOF decomposition and joint-angle divergence (§5.1); "redundant-configuration drift," not "null-space drift" — and, after a correction prompted by external review, not supported by a null-space-projection measurement at all: $E_N$ was found to measure schedule-vs-state Jacobian mismatch, not the control law's own null-space behaviour (§5.2), since a damped-least-squares command is mathematically guaranteed to have zero null-space component against its own Jacobian; the wrong Jacobian is separately shown to be a quantified lag-amplifier ($k_\parallel$ down $\approx30\%$ post-contact, §5.4.3) but not the explanation for the shared failure mode, which both Jacobian pairings exhibit identically; the $q_1$-causal link and the $K_p$ trade-off are both now directly demonstrated via same-state/gradient methods, not only inferred (§5.4.6) |
| **H4** — scheduled-Jacobian accuracy against the contact model at the measured state | **Supported, and shown to precede the H2 failure temporally** | mismatch/gain-ratio/$\epsilon_u$ all elevated in matched 45-55mm and 55-60mm bins, before the 60-64mm failure bin; the headline 60-64mm gain ratio (7.94$\times$) and the earlier-bin elevation both confirmed robust to a denominator-robustness check (§6.6) | Direct re-binned measurement, now checked for statistical robustness; a temporal-ordering observation consistent with H2's causal chain, but — since H2's own same-state directional mechanism is inconclusive (§3.3) — not itself sufficient to establish causality |
| **H5** — the trajectory-varying Jacobian outperformed both tested fixed linearizations | **Supported on outcome; mechanism unresolved** | completion/tracking outcome (scheduled 3/3 + best tracking; idx0 3/3 but worse tracking; idx100 0/3, fails late) is a direct hardware observation | Outcome directly demonstrated; the *why* (a same-state counterfactual was built specifically to test this) is attempted and inconclusive once the metric's sign is corrected (§7.3) — only the prior report's correlational directional-alignment check remains as supporting evidence for the mechanism |
| Proportional-gain ablation (secondary) | **Hypothesis contradicted** | 0% infeasible-request rate at both gains | Unchanged; underdamping is plausible, not demonstrated |

---

## Audit

This document has now gone through three passes: an initial mechanism-
closure pass (against the original `STAGE3_FINAL_REPORT.md`), a second,
corrective pass responding to an external review of that first pass, and a
third pass adding a new analysis (§5.4, the origin of inverse-controller lag
and redundant drift) requested by that same reviewer after reading the
second pass. All three are logged here together; nothing from any pass is
hidden.

**New scripts** (`scripts/`), by pass:

*Pass 1:* `mpc_same_state_replay.py` (shared same-state MPC replay
infrastructure, validated to 0.01% median command-reproduction error),
`h2_workspace_timeline.py`, `h2_same_state_counterfactual.py`,
`h3b_workspace_and_nullspace.py`, `h4_progress_binned.py`,
`h5_same_state_counterfactual.py`, `transient_57mm_part2.py` (+
`transient_57mm_part2_figure_final.py`, `transient_peak_refine.py`).

*Pass 2 (this review response):* `h2_signed_and_schedule_counterfactual.py`
(literal-schedule-window counterfactual + signed $\Delta e$ metric for H2),
`h5_signed_error_reduction.py` (signed $\Delta e$ metric for H5),
`h3b_pipeline_stage_nullspace.py` (raw/clipped/gated $E_N$ decomposition),
`transient_exact_peak_closure.py` (exact-peak-tick same-state replay),
`h4_gain_ratio_robustness.py` (median/aggregate/outlier-excluded gain-ratio
statistics, §6.6).

*Pass 3 (new analysis, same reviewer):* `lag_chronology_analysis.py`
(full-trajectory $e_{lag}/e_\perp/\|e\|/\|u_{raw}\|/E_N/q_1/q_2/h_{x_{max}}$
chronology, §5.4.1-5.4.2), `effective_gain_and_baseline_analysis.py` +
`effective_gain_part23_only.py` (the first has a known bug in its Part 2
only, superseded by the second for Parts 2-3; Part 1's $k_\parallel/k_\perp$
output from the first script is unaffected and was independently verified,
§5.4.3-5.4.5).

**New tables**, by pass:

*Pass 1:* `h2_workspace_margin_timeline.csv`, `h2_workspace_margin_at_marks.csv`,
`h2_same_state_counterfactual.csv` (+ sanity check), `h3b_workspace_failure_decomposition.csv`,
`h3b_workspace_failure_summary.csv`, `h3b_null_redundant_projection.csv`,
`h4_common_progress_bins.csv`, `h5_same_state_counterfactual.csv` (+ sanity check),
`transient_57mm_diagnostic_summary.csv`, `transient_57mm_1D_contact_state.csv`,
`transient_peak_refine.csv`.

*Pass 2:* `h2_signed_and_schedule_counterfactual.csv`,
`h5_signed_error_reduction.csv`, `h3b_pipeline_stage_nullspace.csv`,
`transient_exact_peak_closure.csv`, `h4_gain_ratio_robustness.csv` (+
`h4_gain_ratio_robustness_per_sample.csv`).

*Pass 3:* `lag_chronology_per_tick.csv`, `lag_chronology_EN_strided.csv`,
`lag_chronology_onset_summary.csv`, `k_parallel_k_perp_selective.csv`,
`gain_ablation_lag_reanalysis.csv`, `excess_lag_baseline.csv`.

*Pass 6:* `vessel_geometry_xy_mm.csv`, `tip_vs_vessel_geometry_255mm.csv`.

*Pass 7:* `h3b_margin_rate_decomposition.csv`,
`h3b_kp_counterfactual_margin_rate.csv`.

**New figures**, by pass:

*Pass 1:* `h2_workspace_margin_timeline.png`, `h2_same_state_counterfactual.png`,
`h3b_workspace_drift_mechanism.png`, `h3b_redundant_projection.png`,
`h4_progress_binned_accuracy.png`, `h5_same_state_frozen_counterfactual.png`,
`transient_57mm_deep_diagnostic.png`.

*Pass 2:* `h2_signed_and_schedule_counterfactual.png`,
`h5_signed_error_reduction.png`, `h3b_pipeline_stage_nullspace.png`,
`h4_gain_ratio_robustness.png`.

*Pass 3:* `lag_chronology_combined.png`, `k_parallel_vs_s.png`.

*Pass 6:* `tip_vs_vessel_geometry_255mm.png`.

*Pass 7:* `h3b_margin_rate_chronology.png`, `h3b_kp_counterfactual_margin_rate.png`.

*Pass 4 (requested after reading pass 3, same section):*
`scripts/lag_three_layer_synthesis.py` and
`figures/lag_three_layer_synthesis.png` — a single assembled view of §5.4's
three layers (lag floor, gain reduction, redundant-drift-to-exit), built
entirely from pass 3's own tables with no new live-model evaluation.

*Pass 5 (§5.2 correction, prompted by a further review of pass 4):*
`scripts/verify_null_space_projection_jacobian.py` — the direct check
confirming $E_N$ was computed with $P_N$ built from a different Jacobian
than the one the command was solved with (§5.2).

*Pass 6 (requested after reading pass 5):* `scripts/tip_vs_vessel_geometry_255mm.py`
— the beam tip's own trajectory plotted against the vessel's reconstructed
centreline/wall geometry for the 255mm MPC-$J_C$-vs-MPC-$J_{NC}$ comparison
(§3.2).

*Pass 7 (requested after reading pass 6):* `scripts/h3b_margin_rate_decomposition.py`
— the per-tick, gradient-verified margin-rate decomposition and the
same-state $K_p=0.6$-vs-$1.0$ counterfactual at the actual 255mm
selective-gate failure condition (§5.4.6), directly closing the gain
question §5.4.5 had left open at a different (210mm, hold-gate) condition.

**Claims strengthened in pass 1** (from correlational/outcome-level to
direct same-state or re-binned demonstration — some since revised again in
pass 2, see below): H4's relevance to H2, via temporal ordering (§6.5);
H3b's mechanism, via exact workspace face and two-stage per-DOF account
(§5.1).

**Claims strengthened in pass 2** (confirmed to survive scrutiny, or
genuinely new evidence added): H3b's $E_N$ growth was shown, via a
pipeline-stage decomposition, to be present in the raw pre-clip command and
not injected by downstream clipping/gating (§5.2) — this specific,
narrower finding survives; the broader claim it was reported alongside at
the time ("intrinsic to the controller's own task-space solve," implying
the control law itself generates null-space motion) did **not** survive a
later correction — see "Claims weakened or withdrawn" below, this is not
double-counted as a strengthening. The $s\approx60$-$61$mm local transient
moved from "remains unexplained" to
a candidate mechanism identified (schedule-vs-state Jacobian mismatch at
the exact peak tick, confirmed by both matrix norm and re-solved command)
— §9.1, a genuine new finding, not merely a reworded old one. H4's §6.5
temporal-ordering claim (gain ratio elevated before the terminal tracking
blow-up) is now confirmed robust to a denominator-robustness check across
all three bins, after diagnosing and excluding one anomalous single-tick
live-model evaluation that had initially made the 45-55mm bin look
inconsistent (§6.6).

**New analysis in pass 3** (genuinely new, not a correction of a prior
claim): §5.4's full-trajectory chronology and effective-gain analysis.
Headline new results: a staged ($\approx25$mm), well-ordered
lag$\to$command$\to$redundancy$\to$drift$\to$exit cascade for the
no-contact pairing (compressed but same-ordered for contact); the wrong
Jacobian quantified as a $\approx30\%$ realized-longitudinal-gain reduction
specific to the post-contact region ($k_\parallel$, §5.4.3); a $7$-$24\times$
excess-lag gap between pairings post-contact, consistent with that gain
reduction (§5.4.4); and a re-reading of the existing gain ablation (§8)
showing raising $K_p$ trades modest lag improvement for a *worse* lag tail
and substantially worse cross-track, not a clean lag-for-oscillation trade
(§5.4.5). One data-quality note surfaced during verification, not left
implicit: the gain-ablation runs' final-tick lag value settles into a
`terminal_hold` artefact (an identical $0.0628$mm value recurring across
ticks and across runs, reflecting the terminal reference point's own
polyline-projection geometry, not the gain being tested) — this statistic
is excluded from §5.4.5's reported comparison for exactly this reason.

**Claims weakened or withdrawn in pass 1** (vs. the original report): H2's
prior "retreats to configurations where local tracking is easier" /
workspace-pressure framing — directly contradicted by measured margins
staying generous throughout (§3.2); the prior report's "joint 0 diverges"
single-DOF account of H3b's workspace exit — refined into a two-stage,
multi-joint account (§5.1); "the four redundant directions are completely
free to drift" — replaced with a measured, bounded, growing null-space-
energy fraction (§5.2).

**Claims weakened or withdrawn in pass 2** (vs. pass 1 of *this* document —
the most important entries in this audit): pass 1's H2 §3.3/§3.4 claim that
a same-state counterfactual "directly" showed the actual command realizing
"3-6$\times$ less tip correction" than a contact-informed one is
**withdrawn**: the metric's sign was discarded without justification, and
correcting it reverses rather than confirms the claim. Pass 1's H5 §7.3
claim that the scheduled command realizes "50-200$\times$ more correction"
and that idx0 "directly" outperforms idx100 late in the path is likewise
**withdrawn** for the same reason. Both are downgraded to attempted-and-
inconclusive pending a full delay/horizon-aware cost metric ($\Delta
V_{track}$), not merely reworded with softer language — the quantitative
claims themselves no longer stand. §3.2's pass-1 wording ("a pure
control/tracking-authority effect, not a configuration-space availability
effect") over-corrected the original report and is narrowed. The integrated
discussion's claim that MPC's cost structure "directly avoids" the
redundant-drift failure mode is downgraded to "consistent with," since no
ablation isolates those cost terms as causal. Three numerical/terminology
slips from pass 1 are corrected: H1's "4/4" (should read 2/2 contact-aware
vs 0/2 no-contact, 4 runs total), H5's "any single fixed" overclaim
(neither tested fixed linearization, not every possible one), and an
$s$-vs-insertion-length mix-up in §6.5 ($\approx90$mm is insertion length
$L$, not the $75.3$mm path-progress $s$ actually being discussed).

**New analysis in pass 6** (genuinely new, not a correction): the beam
tip's own measured trajectory plotted directly against the vessel's
reconstructed centreline and wall geometry (§3.2), making H2's completion
asymmetry visually immediate rather than only statistical.

**New analysis in pass 7, the strongest causal closure in this document**:
§5.4.6's margin-rate gradient decomposition (verified against ground truth,
$r=0.911$) directly measures, rather than infers, that $q_1$ dominates the
command's margin-reducing direction by $5$-$9\times$ every other joint in
the failure window; and its same-state $K_p=0.6$-vs-$1.0$ counterfactual,
run at the actual 255mm selective-gate failure condition (not a different
condition as the generic gain ablation was), directly demonstrates that
higher gain makes the per-tick margin step worse at 98.8% of sampled
ticks. This closes, with direct same-state evidence, the one link in the
H3b mechanism that §5.4.5 could previously only address indirectly.

**Claims weakened or withdrawn, §5.2 correction (the single most important
correction in this document — identified after pass 3/4's new analyses had
already been written on top of the flawed premise, so its reach is wide):**
pass 1/2's characterization of $E_N$ (§5.2) as measuring "unregulated
null-space motion" generated by the inverse-Jacobian control law's own
damped-least-squares solve is **withdrawn as mathematically impossible**:
such a command lies exactly in the row space of whichever Jacobian it was
solved with, for any damping value, so projecting it onto the null space of
that *same* Jacobian is guaranteed near-zero — verified directly
($4.7\times10^{-10}$ to $1.1\times10^{-4}$ across 3 sample ticks). The
original analysis instead projected the schedule-solved command onto the
null space of the *live, state-recomputed* contact model — a different
matrix — so $E_N$ actually measures schedule-vs-state Jacobian mismatch,
channelled through the command direction. This is corrected in place in
§5.2, and every downstream reference in §5.4 (which was built using $E_N$
as a chronology quantity, before this correction), §10, and §12 is updated
to describe $E_N$ this way. **Not withdrawn**: the underlying data ($E_N$'s
values, growth, and temporal correlation with margin collapse) remains
valid and reported; "accumulated redundant-configuration drift" as a named
phenomenon survives on §5.1's direct joint-angle evidence alone, which was
never affected by this issue. Also addressed in the same pass, prompted by
the same review: §5.4.4's theoretical lag-baseline formula
($v_s\Delta t/K_p$) was verified, not corrected — reading the controller's
actual `solve()` source confirms this is the physically correct discrete
steady-state balance for *this* controller's specific (incremental,
no-feedforward) control law, so the "7-24$\times$ excess lag" finding
stands as reported; and §5.4.5's gain-ablation conclusion is narrowed to
avoid implying the $K_p=0.6$-vs-$1.0$ test (210mm, hold gate, no-contact
only) directly addresses the 255mm selective-gate failure it sits next to
in the document — it does not, and the text now says so explicitly.

**Mechanisms that remain unresolved**: the full delay/horizon-aware
same-state mechanism for H2 and H5 (the single most important open item —
needs $\Delta V_{track}$, not a one-step metric); a finer characterization
of the transient's $E_J$ anomaly (sharp spike vs. broader plateau, §9.1); the
H5 early-region tracking-quality ordering between idx0 and idx100 (§7.3,
point 3 analogue / §11). The H4 gain-ratio denominator-robustness check
(§6.6) is **resolved**, not open: the headline 7.94$\times$ figure and the
45-55mm/55-60mm temporal-ordering claim both survive, once one diagnosed
single-tick numerical outlier is excluded.

**Would any result here genuinely require a new hardware experiment to
resolve further?** Still no. The remaining open items — the full
$\Delta V_{track}$ computation and the finer transient sweep — are
answerable from existing logs and further offline or same-state
computation. None requires a new closed-loop trial.
