# Study 1 proposal: predictive constraint handling (MPC virtual keep-out vs INV)

**Status: FROZEN DESIGN PROPOSAL. No hardware was run to produce this
document — every number below comes from offline screening against the
already-saved A-E campaign logs and direct, no-hardware calls to the real
MPC worker (`try_solve()`), using the real plan/schedule. This proposal must
be reviewed before any new hardware data is collected.**

## Why this study exists

The just-completed A-E campaign (`proper_research/analysis/condition_A_E_campaign/REPORT.md`)
found that the real in-QP magnet-exclusion constraint (233mm = 230mm reviewed
floor + 3mm robust margin) never came within ~40mm of binding in any of the
21 analyzed runs — minimum measured clearance was 273–280mm throughout. That
campaign could not support or refute any claim about predictive constraints
specifically, because the constraint was never tested. This study proposes a
way to test it, **without changing the real, reviewed 230mm physical safety
boundary.**

## The virtual boundary mechanism

`spawn_and_warm_worker()` (in `run_mpc_delay_aware_insertion_anchor.py`)
already accepts `magnet_exclusion_kwargs=dict(position_jacobians=...,
nominal_positions_m=..., lumen_C_m=..., radius_m=...)` as a plain keyword —
`radius_m` is a simple scalar, independent of the external
`--measured-clearance-abort-mm` check. A second, tighter, **experimental,
in-QP-only** radius can be layered on top of the unchanged physical setup by
passing a different `radius_m` here, with the real 230mm external check
(`common.MAGNET_BEAM_BASE_REVIEWED_BOUNDARY_MM`) left completely untouched.
This is additive, not destructive — no existing flag or default changes.

INV-7's abort behavior is governed *solely* by the same, unchanged 230mm
external check (`--measured-clearance-abort-mm`, confirmed by reading
`run_inv_7dof_delay_aware.py` directly: `magnet_exclusion_radius_m=args.measured_clearance_abort_mm*1e-3`,
no separate in-QP term exists there, because INV has no QP). **Because the
virtual radius lives entirely inside the MPC worker's own QP construction, it
cannot affect INV-7's abort logic at all** — this is a structural guarantee,
not a promise to be careful: whatever radius is chosen for MPC's experimental
layer literally has no code path into INV-7. This directly satisfies "do not
select the radius solely to force an INV failure," since INV's failure
condition is mathematically independent of this choice.

## Offline screening (task 1): choosing the radius from real data, not arbitrarily

Real magnet-to-beam-base clearance, from the already-saved
`predicted_beam_positions.jsonl`/`path_follow.jsonl` of the 21 analyzed
A-E runs (recomputed here, not re-derived from the prior report):

| Condition | Overall min clearance (mm), across all runs |
|---|---|
| A | 272.98 |
| B | 278.24 |
| C (INV, external check only) | **273.36** |
| D | 274.38 |
| E | 273.02 |

**Key finding: C's own minimum clearance (273.36mm) is essentially the same
magnitude as the MPC conditions' (272.98–274.38mm).** All five conditions
track the same reference path, so the magnet's physical trajectory — driven
by the same joint motion needed to follow the same beam path — is nearly
identical regardless of controller. A virtual radius derived from "how close
the real magnet trajectory actually gets" therefore stresses MPC and INV
symmetrically by construction, not just MPC.

**Radius sweep, direct real-worker evidence** (not simulated — real
`spawn_and_warm_worker()` + `try_solve()` calls, no hardware, using 5 real
logged states from run `baseline_A_pilot_2026-11-20_retry1_20261120T122825Z`
at ticks 109–113, the tightest 5 ticks of that run's own measured clearance,
276.60mm, 273.34mm, 273.03mm, 272.98mm, 273.73mm):

| Candidate radius | Worker-reported nominal margin over the reference schedule | Command difference vs 233mm, at the 5 probed real states |
|---|---|---|
| 233mm (production) | 47.0mm | — (baseline) |
| 260mm | 20.0mm | 0.0 at all 5 states — not yet binding |
| 265mm | 15.0mm | 0.0 at all 5 states |
| 270mm | 10.0mm | 0.0 at all 5 states |
| 272mm | 8.0mm | nonzero at 2/5 states (1.6e-3, 3.7e-5) |
| **274mm** | **6.0mm** | **nonzero at 3/5 states (4.3e-3, 1.9e-3, 2.7e-3); 0.0 at the other 2** |

**Chosen virtual radius: 274mm — labeled explicitly as an EXPLORATORY-DATA-SELECTED
candidate, not a pre-registered or independently justified one.** The radius
was chosen by looking at where real logged clearance values from the same 21
runs actually clustered, then testing candidates against that same data until
one showed an effect. That is exploratory selection, not confirmatory
evidence for 274mm specifically — the honest description of what happened,
not a claim that 274mm is somehow the "correct" boundary.

**Explicit selection rule, stated precisely:** among the swept candidates
{260, 265, 270, 272, 274}mm, 274mm is the smallest value at which at least one
of the 5 initially-probed real logged states produced a nonzero predicted-command
difference relative to the production 233mm constraint, while `qp_status`
remained `"solved"` (never `"primal infeasible"`) at every probed state
including the plan's own start. The full sweep table above is preserved in
this document — 260–270mm are reported as genuinely non-binding on this path
at the 5 initially-probed states, not discarded now that 274mm showed an
effect.

**Extended-trajectory update (this section's original 5-state check was
the FIRST pass only — see "Extended check across the available trajectory"
below for the full-coverage result across 1995 real logged states, which
substantially narrows how this effect should be described).**

**Starting clearance (task 1, continued):** at tick 0 of the same real run
(measured clearance 279.98mm, margin 5.98mm over 274mm), a direct
`try_solve()` call with the 274mm virtual radius returns `qp_status==
"solved"` (`tick0_feasibility_check.py`). The plan's own start is feasible
under this virtual boundary.

**Feasible alternative configurations (task 1, continued):** every probed
state (tick 0, and ticks 109–113) returned a genuine `"solved"` QP status at
274mm, not `"primal infeasible"` — the solver found a feasible command at
every one, meaning a recoverable alternative configuration exists near every
tested tight point, not merely that infeasibility is avoided by construction.

**Rate limits (task 1, continued):** no rate limit was changed for this
screening — `joint_velocity_limit_rad_s` (0.10 rad/s) and
`insertion_rate_limit_m_s` (0.002 m/s) are the same production values used
throughout the A-E campaign, and the fact that the QP still finds feasible
solutions under the tighter virtual radius, using these same unchanged rate
limits and the same 15-step horizon, demonstrates the existing rate limits
are compatible with the proposed boundary on this path.

**Nonlinear FK clearance (task 1, continued):** `fk_verification.py`
independently recomputes the magnet position via the real
`urik.forward_kinematics()` + `_resolve_robot_kinematics(make_robot_config())`
(the same functions production code uses, not an approximation) at the 5
tightest logged states, and compares against the logged `magnet_exclusion_gap_mm`.
Agreement is within 0.005–2.3mm at all 5 states (small residual explained by
comparing the commanded joint state `q_cmd_k` against the measured state the
production code actually used for its own gap calculation, not a modeling
discrepancy). The logged clearance values are physically grounded in real
nonlinear FK, not an artifact of the QP's own linearized constraint model.

## Offline screening (task 2): same-state QP ablation, with vs without

`qp_ablation_same_state.py` calls the real worker's `try_solve()` twice from
each of 5 real logged states, once with the production 233mm constraint only,
once with the 274mm virtual constraint added — same state, same schedule,
same horizon, only the constraint radius differs. Raw results:
`qp_ablation_274mm_vs_233mm_results.json`.

**Result (original 5-state pass): the predicted command genuinely differs at
3 of 5 probed states** (||Δu₀|| = 4.3×10⁻³, 1.9×10⁻³, 2.7×10⁻³ — a combined
joint+insertion norm; see the extended pass below for the joint/insertion
split), and is identical at the other 2. **This is mechanistic, offline
evidence that the virtual constraint, at 274mm, measurably changes MPC's
chosen command on this path** — it is a same-state QP re-solve, not a
simulated hardware outcome, and says nothing about what would happen if
actually executed in closed loop (reoptimization at the next tick could
reduce or amplify the effect; this was not simulated further).

### Extended check across the available trajectory (1995 states, all 9 included runs)

`qp_ablation_extended_trajectory.py` repeats the identical with/without
(233mm vs 274mm) `try_solve()` ablation at **every logged tick of all 4
completed A runs and all 5 B runs** (including B's one aborted run, included
explicitly for coverage — its 118 executed ticks are real logged states),
not just the original 5. Raw per-state results:
`qp_ablation_extended_trajectory_results.json`; summary statistics:
`qp_ablation_extended_trajectory_summary.json`.

**This remains offline replay against already-executed states, exactly as
the original 5-state check was — it is not proof that a real closed-loop run
under a 274mm constraint would actually reach, avoid, or recover from any of
these states. Reoptimization under real feedback at the next tick was not
simulated for any of them.**

**Results, reported honestly even though they substantially narrow the
original finding:**

| | Count |
|---|---|
| Total states checked | 1995 (across 10 run directories; the 2 pre-fix-crash directories matched by this glob contributed 0 rows each and are not separately counted) |
| States where 274mm was infeasible or any non-`"solved"` status | **0** — no failures found anywhere in 1995 states |
| States with a nonzero joint-channel command difference | **10 of 1995 (0.5%)** |
| States with a nonzero insertion-channel command difference | **0 of 1995** |
| Max joint-channel delta (rad) | 4.30×10⁻³ (step 109, run `..._122825Z`) |
| p95 joint-channel delta (rad) | 0.0 |

**The effect is real, reproducible, and feasible everywhere it was checked —
but it is far narrower than the original 5-state sample suggested.** All 10
nonzero states come from condition-A runs only (zero from any of the 5 B
runs, despite B contributing 1056 of the 1995 states checked), and all 10
cluster in a narrow 6-sample reference-index window (steps 106–111 of one A
run, steps 109–110 of two others) — one specific region of the path, not a
general property of the trajectory. The 4th completed A run showed zero
nonzero states at all. **Joint vs insertion, reported separately as
requested: the effect is exclusively a joint-channel phenomenon — the
insertion channel never differed, at any of the 1995 states, under either
radius.** This was not visible in the original combined-norm reporting.

**Nonlinear-FK horizon clearance** (`qp_ablation_nonlinear_fk_horizon_results.json`),
computed via real `urik.forward_kinematics()` along the full 15-stage
predicted horizon, for the 20 tightest states found above: the minimum
nonlinear-FK clearance across those 20 states' full horizons was 272.80mm
under the 233mm constraint and 274.98mm under the 274mm virtual constraint —
i.e. the virtual constraint's effect on the chosen command does correspond to
a genuine increase in true nonlinear clearance at the tightest predicted
horizon stages, not merely a change in the QP's own linearized bookkeeping.
Representative example (step 111 of run `..._122825Z`): nonlinear-FK
clearance at horizon stage 0 rises from 273.38mm (233mm constraint) to
275.62mm (274mm constraint) — consistent across the first several horizon
stages at the states checked. This is still a same-state, open-loop
horizon prediction, not a verified closed-loop trajectory.

## The MPC-vs-INV comparison design

- **Shared, unchanged:** plan-dir, schedule-cache, insertion envelope
  (20–70mm), and the external `--measured-clearance-abort-mm 230.0` check —
  identical on both controllers, exactly as in the A-E campaign.
- **MPC-only, new:** the 274mm virtual in-QP radius, added via
  `magnet_exclusion_kwargs` without touching `--beam-base-exclusion-floor-mm`
  (still 230) or `--magnet-exclusion-robust-margin-mm` (still 3.0) — i.e. the
  production 233mm constraint and the 274mm experimental one would need to
  coexist (the tighter one is the one that can actually bind; the simplest
  implementation is to pass `radius_m=0.274` directly, which supersedes
  233mm for this run only, since a single scalar radius is the construction
  the code already supports — not run two constraints simultaneously, which
  would need a small, separate production change outside this proposal's
  scope. **This proposal does not include that code change.** If approved,
  the change needed is: add one new optional CLI flag,
  e.g. `--experimental-magnet-exclusion-radius-mm`, defaulting to `None`
  (current behavior unchanged), that if set, calls
  `spawn_and_warm_worker(..., magnet_exclusion_kwargs=dict(..., radius_m=
  <flag value in m>))` instead of the production 233mm value — purely
  additive, analogous to how `--freeze-jacobian` was added for condition B
  this session.
- **INV-7, unchanged:** no code change needed on the INV-7 side at all — it
  already has no in-QP constraint, and the comparison's "shared external
  intervention rule" is the unchanged 230mm check both controllers already
  use.
- **Not controlled for, stated plainly:** MPC's predictive horizon,
  objective, and redundancy handling differ from INV's reactive law by
  construction — any outcome difference found under this design is still a
  controller-package comparison, as the prior A-E report already learned to
  say for the unconstrained case. This design isolates "does the predictive
  constraint change MPC's own behavior" (yes, demonstrated above, though now
  known to be narrow and localized, not a whole-path effect) and "does that
  visibly change MPC-vs-INV comparative outcomes under a shared external
  rule" (the open empirical question this experiment would answer) — it does
  not and cannot isolate "constraints vs no constraints" as a clean,
  single-factor manipulation between the two controller families, since INV
  structurally has no constraint slot to add one to.

### Intervention semantics: an explicit decision, not left ambiguous

Two designs were considered for what happens when the magnet crosses the
274mm virtual boundary during a fresh hardware run:

- **(i) Observational only.** Log every virtual-boundary crossing as a
  metric; take no control action because of it. The real, reviewed 230mm
  external check (`--measured-clearance-abort-mm`) remains the only thing
  that can ever stop either controller.
- **(ii) A new, separate, shared external stop.** Add a SECOND external
  check (not the real 230mm one) at 274mm, applied identically to MPC and
  INV, that halts the run if crossed. This stop would itself be
  experimental and not yet reviewed as a protective mechanism; the real
  230mm boundary stays as the unchanged, ultimate backstop under this
  design too.

**Decision: this proposal recommends (i), observational only.** The
extended-trajectory check above found the effect to be real but narrow — 10
of 1995 states (0.5%), confined to a single ~6-sample window, joint-channel
only, never affecting the insertion channel, and never infeasible. A design
that stops the run on every 274mm crossing would, on this path, very
plausibly fire rarely or not at all for either controller — adding a new,
unreviewed intervention mechanism for an effect this narrow is a cost (engineering
complexity, a new not-yet-reviewed stop condition touching a real hardware
run) without a correspondingly strong expected benefit in data quality. (ii)
is kept as a documented alternative, not discarded, for a reviewer who wants
direct intervention-based testing instead — if chosen, it must use the exact
same 274mm value and the identical compliance criterion below for both
controllers.

### Identical virtual-compliance criterion for MPC and INV

Whichever design is used, ONE check, computed and reported the SAME way for
both controllers: **the magnet's real measured position (the same
`magnet_xyz_m`/FK-derived position both controllers already log) came within
274mm of the beam base at this tick**, using the same tick-alignment
convention (the shared `step` join key, as used throughout the A-E campaign
analysis) and the same measured-clearance computation already shared by both
controllers for the real 230mm check. MPC additionally has an in-QP term at
this radius; INV does not and never will, structurally — but the
*compliance criterion being scored* is identical for both, so a reviewer can
see this comparison does not use a looser or stricter standard for one
controller than the other.

## The one concrete proposed experiment

**Hypothesis, updated after the extended-trajectory check (narrower than the
original 5-state version, reported honestly rather than preserved as
originally stated):** under a 274mm experimental in-QP keep-out radius (MPC
only; INV's external check stays at 230mm, unchanged, on both), MPC's
predictive constraint will measurably alter its commanded JOINT path —
specifically, not its insertion path, which showed zero sensitivity at this
radius across all 1995 states checked — within a narrow window around
reference-sample index ≈106–111 (the exact window where all 10 nonzero
states in the extended check occurred), while elsewhere along the path MPC's
command is expected to be indistinguishable from the unconstrained case. INV
— lacking this in-QP constraint — is expected to track the unconstrained
reference through that same window and rely solely on the shared external
check if it ever comes within 230mm (which, per the campaign-wide clearance
data above, it has never come close to on this specific path). **This
experiment is now understood, before any data collection, to be testing a
narrow, localized, joint-only effect, not a path-wide one** — a null or
very small result in the tight window itself would be a meaningful outcome
given how small the offline-predicted effect already is (max 4.3×10⁻³ rad).

**Primary metrics:**
1. Minimum measured magnet-to-beam-base clearance, per run (same field as the
   A-E campaign, `magnet_exclusion_gap_mm`/`external_measured_clearance_gap_m`).
2. Active-motion planar tracking RMSE/final error near the identified tight
   window specifically (reference-sample index 103–114, tightly around the
   confirmed 106–111 window, not the whole-path average and not the wider
   95–125 window originally proposed before the extended check — isolating
   the region where the virtual constraint is now known to matter, per the
   extended screening, rather than diluting the already-small effect across
   a wider window or the full 214-sample path).
3. QP `qp_status` per tick for the MPC runs (confirm `"solved"`, never
   `"primal infeasible"`, validating the offline feasibility screening above
   actually holds in closed loop, not just in the single-shot same-state
   checks done here).
4. External-check trigger count/clearance-abort events for both controllers
   (should be zero on both, per the campaign-wide margins found above — if
   this is violated, that is itself a significant, reportable finding).

**Offline evidence supporting this design:** all of the above — the full
radius sweep, the original 5-state ablation, the extended 1995-state
ablation (0 infeasibilities, 10 nonzero states, all joint-channel, all
within one 6-sample window), tick-0 and tightest-tick feasibility, and
independent nonlinear-FK clearance verification at the horizon-stage level.

**Execution order (balanced/randomized, correcting the A-E campaign's
accidental always-same-order confound):** this experiment needs only 2
conditions — "MPC + 274mm virtual constraint" (M) and "INV + unchanged 230mm
external check only" (I) — run n≥4 times each, as a minimum. Order must be
balanced or randomized, not fixed, e.g. an ABBA/BAAB-style balanced sequence
across repetitions such as M, I, I, M, M, I, I, M (or a true random
permutation per repetition, drawn and frozen before data collection, not
adjusted afterward) — never M-always-first or I-always-first across all
repetitions, which is exactly the confound the just-completed A-E campaign's
own report flagged as its most important limitation.

**Explicitly frozen before data collection:** the 274mm radius (an
exploratory-data-selected candidate, not a pre-registered one — stated
plainly, not hidden), the 103–114 reference-index analysis window (updated
from the extended check, narrower than the original 95–125), the four
primary metrics above, the recommended observational-only (design (i))
intervention semantics, the identical MPC/INV compliance criterion, and the
balanced-order requirement are all fixed by this document. Any deviation
found necessary after seeing new data should be reported as a deviation, not
silently incorporated as if it were the original plan.

## What this proposal does not establish

- It does not establish that MPC's predictive constraint is "better" in any
  general sense — only that, at an exploratory-data-selected radius, the
  constraint measurably changes MPC's offline command, and only within a
  narrow, joint-channel-only window covering 10 of 1995 checked states. The
  original 5-state screening somewhat overstated how broad this effect is;
  the extended 1995-state check is the more representative characterization
  and is what this proposal's frozen choices are now based on. Whether any
  of this translates to a meaningful hardware-tracking or safety difference
  against INV is the open question a fresh experiment is designed to answer,
  not an answer in itself.
- It does not add, remove, or relax any approved physical safety boundary —
  the 230mm external check is unchanged throughout, on both controllers, in
  this proposal, under either intervention design considered.
- No hardware has been run under this proposal, including for this extended
  check. All results above, original and extended, come from real logged
  states and real offline worker calls only — this is replay evidence, not
  proof of future closed-loop feasibility.
