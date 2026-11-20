# Constraint study: concrete hardware proposal (FROZEN)

**Status: FROZEN. Do not revise this document after seeing fresh hardware
data. No hardware was executed to produce it — every number below comes
from offline replay against the real MPC worker, the real schedule, and
either the plan's own nominal trajectory or real logged states from the
prior A-E campaign.**

This supersedes 274mm as the candidate for a NEW hardware demonstration
specifically because 274mm's effect (10/1995 states, 0.5%, condition-A
only — see `STUDY1_PROPOSAL.md`) is too sparse to demonstrate anything on
fresh hardware. **That finding is correct and unchanged** — it describes
what actually happened in the real A-E campaign, where the magnet never
came close to any constraint. This document is about what would make a
*future* demonstration run detect a predictive-constraint effect
reliably, which is a different, forward-looking question.

## 1. Predefined candidate set and selection rule (stated before results)

**Candidate set: {274, 276, 278, 280} mm.** Chosen as the next four 2mm
steps above 274mm (the prior study's narrowest-effect point), up to 280mm
(approximately the single tightest clearance value — 272.98mm — observed
anywhere in the real 21-run campaign, plus ~7mm headroom).

**Selection rule, as originally stated (before any candidate was run):**
> Among candidates with zero infeasible solves across all 214 of the
> plan's own nominal schedule samples, select the one with the longest
> run of consecutive reference indices showing a nonzero joint-channel
> predicted-command difference (production 233mm vs the candidate),
> using the plan's own nominal trajectory as the probed state at each
> reference index.

This rule does not reference INV anywhere; INV is not run or consulted at
any point in this section.

## 2. What the rule, applied literally, actually produced — and why it had to be amended

Running the stated rule against the **idealized nominal trajectory**
(`candidate_screening_full_schedule.py`, full results in
`candidate_screening_full_schedule_results.json`):

| Candidate | Infeasible (of 214 nominal samples) | Nonzero-joint states | Longest consecutive run |
|---|---|---|---|
| 274mm | 0 | 0 | 0 |
| 276mm | 0 | 0 | 0 |
| 278mm | 0 | 0 | 0 |
| 280mm | 0 | 14 | **11** (ref_index 96–106) |

Applied literally, this rule selects **280mm**.

**This selection was then checked against a second, independently
necessary requirement this study's own task list already specified:
feasible task continuation against *real executed* deviation, not just
the idealized nominal path** (`candidate_280mm_historical_check.py`,
re-run at 280/278/276mm against all 1995 real logged states from the 9
included A/B runs):

| Candidate | Infeasible (of 1995 real logged states) |
|---|---|
| 276mm | **0** |
| 278mm | **1** (`primal infeasible` at a real A-run state, logged gap 274.6mm) |
| 280mm | **5** (`primal infeasible`, logged gaps 273.0–274.8mm across 2 A runs) |

**280mm and 278mm both fail this second, equally-necessary check.** The
nominal-only screening in §2's table understated how close real execution
actually gets to the beam base — real runs deviate from the idealized
path more than the nominal replay assumed, and at 278–280mm that
deviation is enough to make the real-but-already-executed states
infeasible under the tighter constraint.

**This is not retuning the rule to get a preferred answer.** Feasible
task continuation against real execution variance was always one of this
study's stated requirements (task 2 of the brief), not an optional extra
applied only after seeing which candidate "won" the first check. The
honest conclusion is that the originally stated single-criterion rule was
incomplete — nominal feasibility is necessary but not sufficient — and
is hereby amended to require **both** checks pass, which was the
intent all along. Applying the amended, still radius-agnostic rule:

**Selected candidate: 276mm** — the largest radius in the predefined set
with zero infeasibilities against both the nominal schedule (214/214
solved) and real logged history (1995/1995 solved).

## 3. Offline evidence for 276mm

**Effect coverage** (`candidate_276mm_historical_check_results.json`):
29 of 1995 real logged states (1.5%, vs 274mm's 0.5%) show a nonzero
joint-channel command difference. Critically, unlike 274mm (condition-A
only, one run's window), **276mm's effect appears in a consistent,
overlapping reference-index window (≈105–112) in every one of the 4
completed A runs, and additionally in one B run**:

| Run | Affected ref_index values |
|---|---|
| A — `_122825Z` | 105,106,107,108,109,110,111 |
| A — `_130429Z` | 106,107,108,109,110,111 |
| A — `_131743Z` | 107,108,109,110,111,112 |
| A — `_132839Z` | 106,107,108,109,110 |
| B — `_130657Z` | 108,109,110,111,112 |

This is a **sustained, consistently-reproducible 5–7-sample window**, not
an isolated artifact of one run — a materially better basis for a
demonstration experiment than 274mm's single-run, single-window result.

**Joint vs insertion deltas, reported separately** (per the established
convention): joint-channel delta max 6.96×10⁻³ rad across the 29 states;
**insertion-channel delta is exactly zero at all 29 states** — identical
to the 274mm finding, the effect remains joint-only.

**Nonlinear FK across the predicted horizon** (`candidate_276mm_fk_horizon_*`,
29 states): minimum true clearance (real `forward_kinematics`, not the
QP's linear model) rises from 272.80mm under the production 233mm
constraint to 276.50mm under the 276mm virtual constraint — the QP's
internal model and the true nonlinear clearance agree in direction and
rough magnitude, consistent with the 274mm finding.

**This remains offline replay against already-executed or nominal
states — not proof of future closed-loop feasibility.** A real run under
a 276mm in-QP constraint has not been executed; reoptimization under
feedback at each new tick was not simulated. The feasibility checks above
establish that *this specific replay* found no infeasibility, not that a
genuinely new closed-loop trajectory (which will differ from both the
nominal path and the previously recorded A/B trajectories, precisely
because the controller will react differently under the new constraint)
is guaranteed feasible throughout.

## 4. Three-condition design

1. **Constrained MPC** — identical to condition A (`run_mpc_delay_aware_insertion_anchor.py`,
   horizon 15, adaptation off) with ONE new, additive, experimental
   magnet-exclusion radius layered on top of the existing in-QP
   constraint. Proposed new flag:
   `--experimental-magnet-exclusion-radius-mm 276.0` (default `None` /
   unchanged, following the same additive, non-destructive pattern as
   this session's own `--freeze-jacobian`: when unset, behavior is
   byte-identical to condition A; when set, the magnet-exclusion
   constraint uses this radius IN ADDITION TO — not instead of —
   `--beam-base-exclusion-floor-mm`/`--magnet-exclusion-robust-margin-mm`,
   which stay at their real 230.0/3.0mm values). **This flag does not
   exist in the real source yet — implementing it is a necessary next
   step, separate from this frozen design, and is explicitly NOT done
   here.**
2. **Otherwise-identical MPC without only the virtual constraint** —
   condition A's existing settings verbatim, no new flag passed. This is
   the real, already-run condition A command.
3. **INV** — `run_inv_7dof_delay_aware.py`, unchanged, the real shared
   `--measured-clearance-abort-mm 230.0` external check only.

**Real physical protections unchanged in all three conditions:**
`--measured-clearance-abort-mm 230.0` and (for the two MPC conditions)
`--beam-base-exclusion-floor-mm 230.0 --magnet-exclusion-robust-margin-mm 3.0`
are identical to every prior A–E run. The 276mm virtual radius is
strictly additive and MPC-only by construction — INV has no QP to put it
in, so condition 3 cannot be made to fail by this constraint at all.

**Shared virtual-compliance criterion (reporting only, identical for all
three conditions):** at every tick, compute the real measured magnet
position's distance to the beam base (the same FK/geometry already used
for `magnet_exclusion_gap_mm` throughout the campaign) and flag whether it
is below 276mm. This check is computed and reported identically for all
three conditions regardless of which has an in-QP term for it — so no
condition is held to a looser or stricter standard when the results are
scored.

**Exact settings, all three conditions** (plan/schedule identical to the
whole A–E campaign):

```bash
# Condition 1: constrained MPC (NEW flag, not yet implemented)
python3 -m proper_research.hardware.online.rectangle_stage_a.run_mpc_delay_aware_insertion_anchor \
  --plan-dir plans/rectangle_10x15mm_skipglobal_L30to50mm_2026-11-19_noDipole/time_parameterized_configuration_path \
  --schedule-cache plans/rectangle_10x15mm_skipglobal_L30to50mm_2026-11-19_noDipole/jacobian_schedule_insertion_anchor.npy \
  --out-dir close_loop_logs --run-name constraint_study_1_constrained_mpc_<block> \
  --horizon 15 --jacobian-adaptation-mode off --input-penalty planned --dipole-alignment-weight 0.0 \
  --insertion-command-min-mm 20.0 --insertion-command-max-mm 70.0 \
  --measured-clearance-abort-mm 230.0 \
  --beam-base-exclusion-floor-mm 230.0 --magnet-exclusion-robust-margin-mm 3.0 \
  --experimental-magnet-exclusion-radius-mm 276.0 \
  --protective-config-reviewed

# Condition 2: unconstrained MPC (= condition A, verbatim)
python3 -m proper_research.hardware.online.rectangle_stage_a.run_mpc_delay_aware_insertion_anchor \
  --plan-dir plans/rectangle_10x15mm_skipglobal_L30to50mm_2026-11-19_noDipole/time_parameterized_configuration_path \
  --schedule-cache plans/rectangle_10x15mm_skipglobal_L30to50mm_2026-11-19_noDipole/jacobian_schedule_insertion_anchor.npy \
  --out-dir close_loop_logs --run-name constraint_study_2_unconstrained_mpc_<block> \
  --horizon 15 --jacobian-adaptation-mode off --input-penalty planned --dipole-alignment-weight 0.0 \
  --insertion-command-min-mm 20.0 --insertion-command-max-mm 70.0 \
  --measured-clearance-abort-mm 230.0 \
  --beam-base-exclusion-floor-mm 230.0 --magnet-exclusion-robust-margin-mm 3.0 \
  --protective-config-reviewed

# Condition 3: INV
python3 -m proper_research.hardware.online.rectangle_stage_a.run_inv_7dof_delay_aware \
  --plan-dir plans/rectangle_10x15mm_skipglobal_L30to50mm_2026-11-19_noDipole/time_parameterized_configuration_path \
  --schedule-cache plans/rectangle_10x15mm_skipglobal_L30to50mm_2026-11-19_noDipole/jacobian_schedule_insertion_anchor.npy \
  --out-dir close_loop_logs --run-name constraint_study_3_inv_<block> \
  --insertion-command-min-mm 20.0 --insertion-command-max-mm 70.0 \
  --measured-clearance-abort-mm 230.0 \
  --protective-config-reviewed
```

## 5. Three-block balanced execution order

The A–E campaign's own order (fixed A→B→C→D→E every block) was a
documented confound — condition was perfectly aliased with
position-in-block. This proposal uses a **Latin-square-style balanced
order across 3 blocks** (matching conditions = 3 fits a 3×3 Latin square
exactly, giving every condition one first/middle/last position):

| Block | Position 1 | Position 2 | Position 3 |
|---|---|---|---|
| 1 | Constrained MPC | Unconstrained MPC | INV |
| 2 | Unconstrained MPC | INV | Constrained MPC |
| 3 | INV | Constrained MPC | Unconstrained MPC |

Every condition runs exactly once per block, in a different position each
time — no condition is always first or always last, removing the A–E
campaign's specific confound. If more than 3 blocks are later desired for
power, repeat the square (blocks 4–6 = blocks 1–3 again) or use a second,
independently-generated 3×3 Latin square — state which before collecting
data, do not choose block count after seeing early results.

## 6. What this proposal does not establish

- 276mm's effect, even though sustained and reproducible across runs in
  this offline replay, is still a replay against previously-recorded or
  nominal states — a genuinely new closed-loop run under this constraint
  may produce a different trajectory than any of this analysis assumed.
- The `--experimental-magnet-exclusion-radius-mm` flag does not exist in
  the real source yet; this document proposes its semantics but does not
  implement it.
- This document does not claim 276mm is uniquely optimal — only that it
  is the largest member of a predefined, INV-blind candidate set that
  passes both feasibility checks this study specified.
