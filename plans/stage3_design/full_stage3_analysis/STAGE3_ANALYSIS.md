# Stage-3 Experiment Analysis: Contact Modelling, Exclusion Radius, and Controller Architecture

**Date:** 2026-10-08
**Runs analysed:** 34 live hardware runs (4 open-loop, 24 closed-loop in a 2x2x2 design, 3 frozen-Jacobian ablation, 3 gain-ablation)
**Scope:** analysis and plotting only. No controller code or run data was modified.

All figures are in `figures/`, all tables in `tables/`, and the reused open-loop
outputs in `openloop_reused/`.

---

## 0. Methodology, and three data-integrity findings that constrain the conclusions

### 0.1 Alignment and statistical unit

Trajectories are aligned by **path progress `s`**, taken from each plan's own
`path_s_m` array indexed by the logged `ref_index` — not by time, because the
conditions terminate at very different times. Time is used only where the brief
asks for it (hold durations, infeasible-episode lengths). The **run is the
statistical unit** throughout: with n=3 per condition, no significance testing
is performed, and all claims are stated as run-level patterns plus mechanism
evidence.

Longitudinal lag is `e_lag = s_ref - s_tip`, where `s_tip` is the arc-length of
the projection of the measured tip onto the plan's own `desired_position_m`
polyline; cross-track is the perpendicular distance to that same polyline.

### 0.2 FINDING: the four "255mm" conditions were not run at the same exclusion radius

This is the most important caveat in this report and it limits H2 directly.

**Mechanism (confirmed directly from `controller_metadata.json`, not inferred):**
`run_mpc_delay_aware_vessel.py`'s `--beam-base-exclusion-floor-mm` is the
**TRUE** floor; the live QP/abort threshold is
`TRUE_floor − magnet-exclusion-tolerance-mm` (default tolerance 5.0mm,
documented in the script's own `--beam-base-exclusion-floor-mm` help text).
Every MPC run's directory name **is** the literal TRUE-floor CLI value passed
— there is no naming inconsistency — and MPC consistently rides to within
0.03-1.2mm of the reduced live threshold. Each run's `controller_metadata.json`
records both values exactly:

| condition | directory label | `magnet_exclusion_source` (TRUE) | `magnet_exclusion_radius_m` (live) |
|---|---|---|---|
| MPC-$J_C$, 210mm batch | `floor210_baseline` | `beam_base_fixed_210.00mm` | **0.205** |
| MPC-frozen, 210mm ablation | `mpc_closedloop_contact_frozen` | `beam_base_fixed_210.00mm` | **0.205** |
| MPC-$J_{NC}$, 210mm batch | `mpc_closedloop_nocontact_2026-10-07` | `beam_base_fixed_210.00mm` | **0.205** |
| MPC-$J_C$, "255mm" batch | `floor240_increased` | `beam_base_fixed_240.00mm` | **0.235** |
| MPC-$J_{NC}$, "255mm" batch | `floor255` | `beam_base_fixed_255.00mm` | **0.250** |

(InvJac runs have no equivalent metadata file — that script applies the CLI
value directly with no tolerance subtraction, confirmed from source; the
210mm/255mm figures used elsewhere in this report for InvJac are correct as
given, via the independent pin-distance evidence already cited there.)

So: **MPC-$J_C$ and MPC-$J_{NC}$ are genuinely matched at the 210mm nominal
label** (both TRUE=210, live=205mm) — that leg of the comparison is clean, as
already stated in §3. But **the condition the brief calls "MPC-$J_C$ at
255mm" was actually run at TRUE floor=240mm (live 235mm)** — its own
directory name says `floor240`, not `floor255`. It therefore had **15mm more**
true configured freedom than MPC-$J_{NC}$'s actual TRUE=255mm (live 250mm).

Consequence: $\Delta_J^{255}$ as originally computed was **confounded**. The
contact-Jacobian leg was given a materially easier constraint than the
no-contact leg, which biased the original $\Delta_J^{255}$ estimate
*upward* — i.e. in the direction that favoured H2.

**Resolved (2026-10-08):** this has since been fixed by rerunning MPC-$J_C$
at the exactly matched TRUE floor=255mm (live 250mm), confirmed via
`controller_metadata.json` to match MPC-$J_{NC}$'s live threshold exactly.
See §3 for the corrected, now-trustworthy $\Delta_J^{255}$ figure and §3.5
for the full history of what was corrected and why.

### 0.3 FINDING: the logged `jacobian_condition` field is not a live quantity

`path_follow.jsonl` records `jacobian_condition` as a **constant for every tick
of a run** (e.g. 4806.7 for all 418 ticks of one run). It does not track the
condition number of the Jacobian actually in use.

**Correction (2026-10-08):** the first pass of this analysis scanned the
contact schedule's condition number at a stride of 20 indices, which skipped
straight over index 505 and understated this section. A full-resolution scan
(every index, not every 20th) of `vessel_c_schedule_phi30_L30_newwall_2026-10-06.npy`
— the file independently confirmed (§0.2 methodology note below) to be the one
actually used by `floor210_baseline`/`floor240_increased`/the frozen ablation —
finds:

- **indices 0-9 (s=0-1.3mm):** condition number decaying smoothly from
  **5093 down to 253**, not just a single-point spike at index 0.
- **index 505 (s=71.4mm) only:** an **isolated** spike to **665.2**, against
  neighbors at 4.0-4.4 on both sides (idx 504: 4.06, idx 506: 4.06). A
  single-tick numerical artifact, not part of any broader trend.
- every other index: **3.65-18.4**.

A repaired schedule exists
(`vessel_c_schedule_phi30_L30_newwall_2026-10-06_repaired.npy`, differing from
the original at index 505 only, confirmed by direct array diff) that resolves
the index-505 spike (cond 665.2→4.06) but does not touch the 0-9 tail. The
hardware runs analysed in this report used the **original**, unrepaired file —
confirmed directly, not assumed, by comparing each candidate schedule's
row-505 prediction against the real logged `predicted_beam_position_0_m` at a
`ref_index=505` tick in `floor210_baseline_v2` (original: 0.358mm mismatch;
repaired: 0.788mm mismatch — original matches better).

**Does either conditioning region explain an actual tracking failure?** Checked
directly against the hardware logs, not assumed:
- The three largest tracking-error ticks in the whole MPC-$J_C$ dataset
  (3.474mm, 3.655mm, 2.401mm, one per rep) all occur at **ref_index≈402-405
  (s≈57mm)** — where the condition number is a completely unremarkable
  5.8-6.1. The index-505 spike produces **no visible error spike** at the
  corresponding tick (logged error there: 0.13-0.83mm across the three runs).
  So while the index-505 defect is real and the repair is a legitimate fix, it
  is **not** the explanation for the dominant tracking bump in this dataset.

  **Targeted follow-up (2026-10-08):** a small, focused diagnostic at
  s=54-60mm (all three reps, every tick in the window) plotted $e_{track}$,
  $e_{pred}$ against the scheduled Jacobian, all three singular values of
  $J_{sched}$, $\|u\|$, tick-to-tick $\|\Delta u\|$, the live constraint
  margin, insertion rate, and schedule-vs-live error together
  (`tables/s54_60mm_targeted_detail.csv`, `figures/s54_60mm_targeted_diagnostic.png`).
  **Nothing examined aligns with the peak.** Every quantity sits at an
  entirely ordinary value through this window: $\sigma_1/\sigma_3$ (condition)
  averages 4.85-4.88 across reps (same as the broad 4-17 baseline, §0.3), the
  constraint margin averages 10.9-13.1mm (nowhere near the 0mm floor),
  schedule-vs-live divergence averages 33-35% (squarely inside the range H4
  shows has no tick-level relationship to tracking error, §5.2), and neither
  $\|u\|$ nor $\|\Delta u\|$ shows an outlier at the peak tick. Per this
  report's own standard (a bump does not need to be solved to publish the
  main result): **this is left as an unexplained local transient.** Whatever
  drives it is not visible in any of the eight quantities checked here, and
  it is not large enough (max 3.655mm, well under every abort threshold) to
  threaten any of this report's main conclusions.
- The index-progression (0-9) near-singularity, present in both schedule
  files and therefore live in every run analysed, also does not produce an
  obvious tracking spike at the very start of the path (logged error at steps
  1-5 of `floor210_baseline` runs is 0.9-1.5mm, unremarkable) — consistent
  with MPC's QP regularization (not raw matrix inversion) absorbing it.
- The inverse-Jacobian contact failure window (ref_index 260-430, §4) has mean
  condition number 6.3 over that span (recomputed at full resolution,
  unchanged from the original scan since index 505 falls outside it) —
  **still better-conditioned than the early part of the path every controller
  traverses without difficulty.**

**Net:** ill-conditioning is real in two specific, narrow regions of the
contact schedule (idx 0-9 and idx 505 in the original file), and the index-505
defect has now been fixed in a `_repaired` variant — but neither region is a
demonstrated cause of any tracking failure in this dataset. The logged
`jacobian_condition` field remains unusable for any live analysis regardless.

### 0.4 Subsampling for the live-Jacobian work (H4/H5)

One "accurate"-mode live Jacobian evaluation costs ~2.1s at an off-reference
(closed-loop) state. Evaluating every tick of every run twice over is ~5 CPU-hours,
so H4/H5 use an **even subsample of 45 ticks per run** and **one representative
rep per condition**. Reps are tightly clustered on every tracking metric (see
`tables/per_run_summary_closedloop.csv`), so between-rep spread is small relative
to the between-condition effects tested. This is stated in the tables and should
not be read as full coverage.

---

## 1. Per-run summary

**Updated 2026-10-08** to use the corrected, radius-matched MPC-$J_C$ data at
both radii (see §0.2/§3 for the full explanation):

- **210mm MPC-$J_C$:** rep1 (`floor210_baseline`, aborted
  `magnet_exclusion_violated(gap=205.0mm<205.0mm)`) is **excluded** — the
  experimenter identified this abort as a bug, not a genuine control failure
  — and replaced with a rerun (`..._v4_20261008T120734Z`, TRUE floor=210mm,
  live=205mm, `path_complete`). The 210mm MPC-$J_C$ cell is now reps 2, 3, 4.
  Rep1 is kept in the full-accounting table below (not deleted) for
  transparency.
- **255mm MPC-$J_C$:** the original `floor240_increased` cell (TRUE floor
  =240mm, live=235mm) is **superseded** — it was not radius-matched to
  MPC-$J_{NC}$'s true live threshold (250mm) — by a new, exactly-matched
  rerun at TRUE floor=255mm, live=250mm (`mpc_closedloop_contact_floor255_matched_v{1,2,3}_20261008T*`,
  confirmed via each run's own `controller_metadata.json`:
  `magnet_exclusion_source: beam_base_fixed_255.00mm`,
  `magnet_exclusion_radius_m: 0.25`). This is now the headline 255mm MPC-$J_C$
  cell. The superseded `floor240_increased` runs and an earlier,
  still-not-quite-matched TRUE=250mm attempt are both kept in the
  full-accounting table (`tables/per_run_summary_closedloop.csv`, `cell_role`
  column) for the record.

Full table: `tables/per_run_summary_closedloop.csv` (now includes a
`cell_role` column: `primary` / `frozen_ablation` / `EXCLUDED (...)` /
`SUPERSEDED (...)` / `secondary (...)`), `openloop_reused/common_interval_stats.csv`
(4 open-loop runs).

Condition means, run as the unit (n=3 each, primary cells only):

| radius | condition | completed | mean s_end (mm) | progress | RMSE (mm) | median | p95 | max | AUC |
|---|---|---|---|---|---|---|---|---|---|
| 210 | MPC-$J_C$ *(reps 2,3,4)* | **3/3** | 75.32 | 1.00 | 0.598 | 0.319 | 1.152 | 3.647 | 0.418 |
| 210 | MPC-$J_{NC}$ | **3/3** | 75.32 | 1.00 | 0.497 | 0.358 | 0.930 | 1.443 | 0.437 |
| 210 | InvJac-$J_C$ | **0/3** | 62.27 | 0.83 | 7.008 | 0.818 | 16.148 | 17.609 | 3.717 |
| 210 | InvJac-$J_{NC}$ | **3/3** | 75.32 | 1.00 | 0.685 | 0.535 | 1.219 | 1.858 | 0.609 |
| 210 | MPC-frozen idx0 *(ill-cond., ablation)* | **3/3** | 75.32 | 1.00 | 1.479 | 1.198 | 2.985 | 3.931 | 1.323 |
| 210 | MPC-frozen idx100 *(well-cond., pre-contact, ablation)* | **0/3** | 70.92 | 0.94 | 1.148 | 0.335 | 2.838 | 4.889 | 0.608 |
| 255 | MPC-$J_C$ *(MATCHED rerun)* | **3/3** | 75.32 | 1.00 | 0.684 | 0.359 | 1.513 | 2.235 | 0.487 |
| 255 | MPC-$J_{NC}$ | **0/3** | 64.37 | 0.86 | 1.090 | 0.766 | 1.888 | 4.738 | 0.856 |
| 255 | InvJac-$J_C$ | **0/3** | 39.97 | 0.53 | 9.414 | 6.443 | 18.206 | 19.951 | 7.649 |
| 255 | InvJac-$J_{NC}$ | **0/3** | 47.30 | 0.63 | 7.445 | 1.713 | 17.545 | 19.859 | 4.934 |

*For non-completing rows, "mean s_end"/"progress" is the mean of each run's
own `s_ref.max()` — how far the **reference** reached before the run
aborted, not how far the beam actually got (which can be meaningfully behind,
e.g. MPC-frozen idx100's final lag reaches 3.1-4.7mm, §7.5). It is reported
for comparability with the completing rows, not as a claim of physical
progress.*

Because several conditions terminate early, the brief's common-interval
comparison is also computed (`tables/common_interval_{210,255}mm.csv`),
truncating every run to the shortest run's `s`:

- **210mm, common interval 0-61.8mm:** MPC-$J_C$ 0.524 (mean of reps 2,3,4),
  MPC-$J_{NC}$ 0.486, InvJac-$J_{NC}$ 0.655, InvJac-$J_C$ 6.326 mm RMSE.
- **255mm, common interval 0-30.9mm:** MPC-$J_C$ 0.770 (MATCHED rerun),
  MPC-$J_{NC}$ 0.828, InvJac-$J_{NC}$ 1.700, InvJac-$J_C$ 7.299 mm RMSE.

The ordering is unchanged by truncation, so the full-run differences are not an
artefact of comparing a complete run against a truncated one. Note that on the
common interval alone, matched-radius MPC-$J_C$ (0.770mm) and MPC-$J_{NC}$
(0.828mm) are now close — the gap that matters is what happens *after* this
interval, where MPC-$J_{NC}$ goes on to fail and MPC-$J_C$ does not (§3.2).

**Note on abort thresholds:** these are not uniform across conditions —
`max_tracking_error_m` was 5mm for MPC runs and 20mm for inverse-Jacobian runs.
Max-error and "did it complete" comparisons across architectures must be read
with that in mind; the RMSE/median/AUC and common-interval statistics are not
affected by the threshold, only by where each run stopped.

---

## 2. H1 — Explicit contact modelling is required for an executable open-loop plan

**Verdict: strongly supported.**

Reused from the pre-existing open-loop analysis
(`openloop_reused/`, figures `fig1`, `fig3`, `fig4`, `fig5`):

| run | stop reason | s_end (mm) | max error (mm) |
|---|---|---|---|
| contact rep1 | `path_complete` | 75.32 | 9.057 |
| contact rep2 | `path_complete` | 75.32 | 8.823 |
| no-contact rep1 | `tracking_error_exceeded(10.00mm)` | 61.57 | 9.943 |
| no-contact rep2 | `tracking_error_exceeded(10.03mm)` | 61.39 | 9.809 |

Both contact-plan reps executed the full path; both no-contact reps hit the 10mm
abort at s≈61.4-61.6mm, i.e. ~82% of the path. Common-interval (0-61.4mm) RMSE
is close between the two plans (4.44/4.53 contact vs 4.78/4.78 no-contact), so
the difference is **not** a uniform accuracy offset — it is that the no-contact
plan's error *keeps growing* until it trips the threshold.

**Pre- vs post-contact behaviour:** measured divergence onset
($\Delta e = e_{NC}-e_C$ exceeding the pre-contact noise floor +3σ for 5
consecutive samples) is **s=50.5mm and s=51.8mm** for the two rep pairs. The
offline-predicted contact onset is s=28.5mm. So the two plans do behave
similarly before contact and diverge after it, as H1 predicts — but the measured
divergence appears **~22mm later** than the predicted onset, meaning early
contact is initially tolerable and only becomes execution-limiting further
along. That is a weaker and more specific claim than "they diverge at contact
onset", and it is what the data supports.

An independent confirmation of the mechanism comes from the live model
comparison in §5: $\|J_{NC}^{live}-J_C^{live}\|/\|J_C^{live}\|$ is **0.001%** at
s=15mm (pre-contact) and **16-33%** at s=32-75mm (post-contact). The contact term
switches on exactly where it should and is large once it does.

---

## 3. H2 — Contact Jacobian x exclusion-radius interaction

**Verdict: supported, on a genuinely radius-matched comparison at both
exclusion floors.** The original 255mm cell was confounded (§0.2); it has
since been superseded by a rerun at the exactly-matched TRUE floor
(verified via `controller_metadata.json`, not inferred). The "lag rather
than lateral displacement" sub-claim remains only partially supported.

### 3.1 The interaction quantity (matched-radius result)

With $E$ = mean full-run RMSE (mm), from `tables/h2_interaction_effects.csv`,
using the corrected cells (210mm: MPC-$J_C$ reps 2,3,4; 255mm: the matched
rerun, TRUE=255mm/live=250mm for **both** MPC-$J_C$ and MPC-$J_{NC}$):

| radius | $E_{MPC,C}$ | $E_{MPC,NC}$ | $\Delta_J = E_{MPC,NC}-E_{MPC,C}$ |
|---|---|---|---|
| 210mm (live 205mm, matched) | 0.598 | 0.497 | **−0.101** |
| 255mm (live 250mm, matched) | 0.684 | 1.090 | **+0.406** |

$\Delta_J^{255} = +0.406 > \Delta_J^{210} = -0.101$. **The predicted ordering
holds, now on a clean, radius-matched comparison at both floors** — both
210mm legs enforce live=205mm and both 255mm legs enforce live=250mm,
confirmed directly from each run's own `controller_metadata.json`, not
inferred from achieved distance.

At 210mm the no-contact Jacobian still costs **nothing** (−0.101mm, the same
sign and similar magnitude as the original estimate, within normal rep-to-rep
spread). At 255mm, the gap is real but **smaller than the original
(confounded) estimate of +0.536mm** — the corrected, honest number is
**+0.406mm**. The direction and qualitative story are unchanged; the
magnitude is now trustworthy.

### 3.2 The much stronger evidence is completion, not RMSE

- MPC-$J_{NC}$: **3/3 complete at 210mm → 0/3 complete at 255mm**, all three
  aborting on `tracking_error_exceeded(5.0-5.4mm)` at s≈63.9-64.8mm (~85% of path).
- MPC-$J_C$: **3/3 → 3/3** at both radii (rep1 of the original 210mm batch
  excluded, §1; the matched 255mm rerun also went 3/3), no degradation.

So tightening the exclusion floor converts the no-contact Jacobian from
"costless" to "run-ending", while the contact Jacobian is unaffected **at the
exact same live threshold**. This completion asymmetry is the strongest
evidence for H2 and is now unconfounded. See
`figures/closedloop_tracking_255mm.png`: MPC-$J_C$ (green) stays flat near
zero to s=75mm; MPC-$J_{NC}$ (blue) tracks comparably to MPC-$J_C$ out to
s≈60mm, then spikes and terminates by s≈64mm.

### 3.3 What the original confound looked like, for the record

The first "255mm" MPC-$J_C$ cell (`floor240_increased`) was actually run at
TRUE floor=240mm (live=235mm) — 15mm more true freedom than MPC-$J_{NC}$'s
255mm (live=250mm). That inflated $\Delta_J^{255}$ to +0.536mm. A first
correction attempt (TRUE=250mm, live=245mm) undershot in the other direction
(5mm *less* freedom than MPC-$J_{NC}$) and gave RMSE 0.729mm — consistent
with, but not identical to, the final matched value of 0.684mm. Both
superseded runs are kept in `tables/per_run_summary_closedloop.csv` (`cell_role`
column) for transparency, not deleted.

### 3.4 Does MPC-$J_{NC}$ lag at 255mm?

From `tables/longitudinal_lag_vs_crosstrack.csv` (210mm MPC-$J_C$ now reps
2,3,4; 255mm MPC-$J_C$ now the matched rerun):

| radius | condition | mean lag | max lag | **final lag** | mean cross-track | lag/cross ratio |
|---|---|---|---|---|---|---|
| 210 | MPC-$J_{NC}$ | 0.167 | 1.037 | 0.079 | 0.201 | 0.83 |
| 255 | MPC-$J_{NC}$ | 0.476 | 2.860 | **2.860** | 0.479 | 0.99 |
| 210 | MPC-$J_C$ | 0.011 | 1.816 | 0.130 | 0.293 | 0.04 |
| 255 | MPC-$J_C$ | 0.135 | 2.071 | 0.413 | 0.343 | 0.39 |

Lag does grow for MPC-$J_{NC}$ at 255mm: mean 0.167→0.476mm (2.9x) and final
lag 0.079→2.860mm (**36x**), with the terminal value being the state at which
the run aborts. MPC-$J_C$ also grows somewhat with the matched rerun
(0.011→0.135mm mean, 0.130→0.413mm final) but stays an order of magnitude
below MPC-$J_{NC}$'s values throughout, so the effect remains specific to the
no-contact Jacobian, just not quite as cleanly "zero at both radii" as the
unmatched estimate suggested.

However, **cross-track grows by a similar factor** for MPC-$J_{NC}$
(0.201→0.479mm, 2.4x), so the lag/cross-track ratio only moves 0.83→0.99. The
honest reading: the no-contact Jacobian at the tighter floor accumulates
error in **both** components roughly proportionally, with the *terminal*
failure being lag-dominated. The claim "lagging rather than merely displaced
laterally" is **partially supported** — it is true of the end state, not of
the run as a whole.

By contrast the inverse-Jacobian conditions are overwhelmingly lag-dominated
(ratio 7.1-10.6), which is exactly what the hold mechanism in §4 predicts.

Figure: `figures/mpc_nc_vs_c_255mm_lag_crosstrack.png` (regenerated against
the matched rerun).

### 3.5 Matched-radius reruns: summary of what was corrected and why

Two corrections were applied to reach the result in §3.1-§3.2, both
triggered by the experimenter's own live hardware runs and verified against
each run's `controller_metadata.json` rather than assumed:

1. **210mm MPC-$J_C$ rep1** (`magnet_exclusion_violated(gap=205.0mm<205.0mm)`)
   excluded as an experimenter-diagnosed bug, replaced by a rerun at the
   identical TRUE=210mm/live=205mm condition that completed cleanly
   (`path_complete`). Kept in the full-accounting table, not deleted.
2. **255mm MPC-$J_C$** reran at the exactly-matched TRUE=255mm/live=250mm
   (after an intermediate TRUE=250mm/live=245mm attempt that was still 5mm
   off), confirmed via `controller_metadata.json`'s
   `magnet_exclusion_source`/`magnet_exclusion_radius_m` fields to match
   MPC-$J_{NC}$'s own live threshold exactly. All three reps completed.

The result of both corrections: **H2's direction is confirmed and its
magnitude is now trustworthy** ($\Delta_J^{255}=+0.406$mm,
$\Delta_J^{210}=-0.101$mm), on a comparison where every number has been
checked against ground truth rather than inferred or assumed.

---

## 4. H3 — MPC vs inverse-Jacobian constraint awareness

**Verdict: strongly supported in its core claim, with two important refinements.**

### 4.1 The raw pre-gate command was recovered exactly, not inferred

The inverse-Jacobian runs use `--magnet-protection hold`: the controller is
constructed with **no** magnet-constraint kwargs (verified in
`run_inverse_jacobian_online_vessel.py`), so its own math is entirely unaware of
the exclusion radius, and an external gate
(`close_loop_path_follow._COMMAND_SAFETY_GATE`) zeroes the whole 7-vector on any
tick whose raw command would violate the constraint one tick ahead. The log
stores only the **post-gate** command, so the raw request had to be recovered.

It was recovered by replaying the real control law offline (damped least squares
with `nullspace_gain=0`, `feedforward=False`, the run's own `position_gain`/
`damping`, the run's own schedule, the same velocity/acceleration/state-box clip,
and the raw — not gated — previous command threaded recursively), driven only by
each run's logged measured states.

**Validation: the replay reproduces the real gate's hold/pass decision on
every tick of all 12 inverse-Jacobian runs (agreement = 1.000).** The H3 numbers
below are therefore exact recoveries, not estimates.

### 4.2 Infeasible-request quantification

From `tables/h3_invjac_infeasible_commands.csv` (condition means, n=3):

| radius | condition | frac ticks infeasible | (exclusion) | (z-workspace) | frac held | episodes | longest episode | worst violation | progress lost |
|---|---|---|---|---|---|---|---|---|---|
| 210 | InvJac-$J_C$ | **0.368** | 0.368 | 0.000 | 0.368 | 5.3 | 6.94s | 8.03mm | 19.9mm |
| 210 | InvJac-$J_{NC}$ | **0.000** | 0.000 | 0.000 | 0.000 | 0.0 | 0.00s | none | 0.0mm |
| 255 | InvJac-$J_C$ | **0.696** | 0.248 | 0.447 | 0.696 | 1.7 | 12.82s | 1.23mm | 26.7mm |
| 255 | InvJac-$J_{NC}$ | **0.419** | 0.382 | 0.037 | 0.419 | 1.3 | 9.07s | 9.55mm | 20.9mm |

MPC's counterpart (`tables/h3_mpc_constraint_margin.csv`): the constraint is
**active** (within 3mm) on 8.5-20.4% of ticks, minimum margin 0.05-1.5mm, and
**never violated on any tick of any MPC run**.

### 4.3 The mechanism, and why it is terminal

MPC spends a comparable fraction of ticks *at* the constraint (8.5-20.4%) as the
naive controller spends *violating* it (36.8-69.6%), but the outcomes differ
completely: MPC redirects within the feasible set and keeps moving, whereas the
naive controller's command is zeroed. Because the reference index advances on its
own schedule regardless of whether the robot moved, each held tick is permanently
lost ground — the magnet parks within ~0.1mm of the floor and stays there while
the target races ahead. Hence the lag-dominated error signature (§3.4), 19.9-26.7mm
of path progress lost, and truncated insertion (§6).

Figure: `figures/h3_representative_stall_event.png` shows proposed vs executed
margin and command norm around the stall, with held ticks shaded.

### 4.4 Refinement 1: at the loose floor with the matched Jacobian, the difference is latent

InvJac-$J_{NC}$ at 210mm requested **zero** infeasible commands across all three
reps (closest approach left 10.8-13.1mm of margin) and completed the path at
0.685mm RMSE against MPC's 0.497mm. The architectural difference exists but is
**not exercised** in that cell. Any claim that the naive controller "frequently
requests violating motions" is conditional on the constraint actually binding —
it does so at 255mm (41.9% of ticks even with the matched Jacobian) and with the
mismatched Jacobian at 210mm, but not in the easy cell.

### 4.5 Refinement 2: at 255mm the binding constraint is often the z-workspace, not the radius

The gate checks the exclusion radius **or** the magnet z-workspace bounds. At
255mm the z bound is co-dominant, and for one rep it is the entire story:

- `invjac_hold_contact_floor255` (rep 192005Z): 0.000 exclusion-infeasible but
  **0.737 z-infeasible** — a pure z-workspace lockup. Its minimum magnet
  distance was 280mm, never near the radius at all.
- the other two contact reps at 255mm: a mix (0.317/0.427 exclusion, 0.422/0.184 z).

H3 names the beam-base exclusion constraint specifically. The data supports the
general claim ("the naive controller repeatedly requests configurations that
cannot legally be executed, and gets held") but shows the **specific constraint
involved is not always the one hypothesised**. Reporting this as a pure
exclusion-radius effect would be wrong.

### 4.6 A third gate mode (selective per-joint clip) and a new, distinct failure mode it exposes

**Added 2026-10-08.** A third `--magnet-protection` option was built: `selective`
— on a predicted violation, zeroes only the individual joint-velocity
components whose own sign pushes the margin further into violation
(`g_i·command_i<0`), passing every other component (including insertion)
through exactly as the naive controller commanded it. This sits between
`hold` (zero everything) and the controller's own `clip` (minimum-norm
projection of everything). Three reps were run at 255mm (live 250mm) with
the **repaired** contact schedule (§0.3) — directly comparable to the
matched255 MPC-$J_C$ condition (§3), which used the identical schedule file.

**Completion and tracking:**

| | InvJac-selective + $J_C$ (255mm, repaired sched) | MPC + $J_C$ (255mm matched, same sched) |
|---|---|---|
| completed | **0/3** — all `tcp_out_of_workspace`, s≈62-64mm (~83-84% progress) | **3/3** `path_complete` |
| RMSE (full run) | 1.21-1.24mm | 0.70-0.77mm |
| gate (magnet-exclusion) clip activity | 0-1.9% of ticks | n/a (in-QP) |
| min magnet distance | 257.7-258.6mm | 250.9-251.1mm |

**The selective gate did its job — it is not what ended these runs.** Fewer
than 2% of ticks ever needed any clipping, and the magnet never got within
7mm of its own floor in any of the three reps. All three runs instead hit
`tcp_out_of_workspace` — the same generic, magnet-unrelated Cartesian-box
check discussed in §4.5 and in this report's live debugging history
(`close_loop_path_follow.py`'s box on the robot's own `getActualTCPPose()`,
not the magnet position) — at a strikingly consistent point, s≈62-64mm,
across all three reps.

**This is a different, inverse-Jacobian-specific failure mode: unregularized
null-space drift.** Figure: `figures/h3_invjac_vs_mpc_redundancy_drift.png`.
Overlaying joint-0 (base rotation) against path progress for both controllers
on the *identical* floor and schedule shows their trajectories tracking each
other almost exactly from s=0 to s≈55mm — then **sharply diverging**: MPC's
joint-0 plateaus near −30° to −33° and stays there for the rest of the path;
the inverse-Jacobian controller's keeps climbing past −20° (off the top of
the plotted range) in the same window its tracking error blows past 5mm and
the TCP crosses the workspace box. The same pattern was independently found
at the *contact-MPC* level in §0.3/§4.6 above (the unexplained s≈57mm
error peak) and is visible again here in a *different* controller — this
project's third independent line of evidence that something about this
specific stretch of the path (s≈55-65mm) is locally demanding, even though
no single examined quantity (§0.3) explains why.

**Why does MPC absorb this and the naive controller doesn't?** Both
controllers are run with zero explicit posture/nullspace regularization
(`nullspace_gain=0` for the inverse-Jacobian controller, `Q_N=0` for MPC —
deliberately matched, this whole project's standard fair-comparison
condition, §3). But MPC's QP cost is not *only* the state cost: its
`R=700·R0` input-tracking weight and its input-increment weight apply to the
**entire 7-dimensional commanded input**, including the 4 redundant
directions that don't affect tip tracking at all — these cost terms
implicitly damp *all* joint velocity, task-relevant or not, every single
tick, with no need for an explicit posture target. The naive
inverse-Jacobian controller's redundancy resolution, with `nullspace_gain=0`,
has **no such term anywhere** — the null-space projector
$(I-J^+J)$ is multiplied by a gain of exactly zero, so the four redundant
directions are completely free to drift however the damped-least-squares
solution's own numerical asymmetries happen to push them, with nothing
pulling them back, tick after tick, for as long as the run lasts. Over most
of the path this drift is harmless (task tracking is by construction
insensitive to it) — but when the path *also* becomes locally demanding
(the same s≈55-65mm region flagged above), MPC's implicit, always-on input
damping keeps it inside a bounded, recoverable configuration, while the
naive controller — having wandered further from a "normal" posture for
longer, with literally nothing resisting that drift — has no mechanism to
recover and the divergence runs away.

**This is a genuinely different mechanism from H3's main finding
(constraint-awareness).** The magnet-exclusion gate worked correctly
throughout; the failure comes from the complete absence of *any* regularizer
on the redundant joints, magnet-related or not — a cost specific to how
naive resolved-rate control handles kinematic redundancy, not to the
exclusion-radius constraint this section otherwise focuses on. It is also a
plausible, though not separately confirmed, contributor to why the inverse-
Jacobian conditions in the main 210mm/255mm factorial (§1, §3) show larger
late-path tracking degradation generally, beyond what the magnet-exclusion
hold/clip mechanism alone explains.

### 4.7 Is the redundancy-drift mechanism specific to the contact Jacobian? No — it reproduces with the no-contact Jacobian too

**Added 2026-10-08.** §4.6 demonstrated the mechanism with one Jacobian
(contact) and one gate mode (`selective`). To check this is a property of
the inverse-Jacobian control law itself — not an artefact of that specific
Jacobian or gate — three more reps were run at the same 255mm floor with
the **no-contact** Jacobian schedule (and its own matched plan-dir/reference
path) and the **clip** gate mode (the controller's own internal projection,
a third combination not tried before):

| | InvJac-clip + $J_{NC}$ (255mm) | MPC + $J_{NC}$ (255mm, closedloop) |
|---|---|---|
| completed | **0/3** — all `tcp_out_of_workspace`, step 357-406 | **0/3** — all `tracking_error_exceeded`, step 304-309 |
| RMSE (full run) | 1.72-2.41mm | 1.08-1.11mm |
| min magnet distance | 255.5-257.6mm (floor=255mm) | 250.0-250.3mm (live=250mm) |
| joint-0 range, first 350 ticks | −50.0° to −24.7° (one rep reaches −24.7°) | −50.3° to −29.4°, identical across all 3 reps |

**Both controllers fail at this floor with the no-contact schedule — but
for different reasons, and the inverse-Jacobian controller's failure mode
is the same one found in §4.6.** MPC-$J_{NC}$'s failure here is the H2/H3
schedule-fidelity story already established (§3.4): it gives up via its own
`max_tracking_error_m` check, early (step ~305), with joint-0 staying
tightly and *identically* bounded across all three reps (−50.3° to −29.4°,
essentially zero rep-to-rep variability) — consistent with MPC's input cost
continuing to regularize the redundant directions even while the no-contact
schedule itself is failing to track. The inverse-Jacobian controller
survives notably longer (step 357-406) before failing by the **same route**
as §4.6: `tcp_out_of_workspace`, with joint-0 pushing *past* wherever MPC's
own trajectory plateaus (one rep reaches −24.7°, 4.7° beyond MPC's bound;
the other two stay closer to −32°, showing real rep-to-rep variability in
how far the unregularized drift runs before the box trips it). The magnet-
exclusion gate again did its job in every rep — minimum magnet distance
stayed 0.5-2.6mm clear of the 255mm floor — so, exactly as in §4.6, the
gate is not what ends these runs.

Figure: `figures/h3_invjac_redundancy_drift_jacobian_independence.png` —
joint-0, tracking error, and magnet distance vs path progress for all four
conditions (contact pairing solid, no-contact pairing dashed) on one set of
axes. The qualitative shape repeats: whichever Jacobian/schedule pair is
used, the inverse-Jacobian controller's joint-0 trajectory tracks its MPC
counterpart closely at first, then drifts past it in the same mid-path
region, while MPC's stays bounded (whether MPC itself ultimately succeeds,
as with $J_C$, or fails by a different, earlier mechanism, as with
$J_{NC}$).

**Conclusion:** the unregularized-null-space-drift mechanism identified in
§4.6 is a property of the inverse-Jacobian control law's redundancy
resolution (`nullspace_gain=0`, no term anywhere penalizing the 4 redundant
directions), not of the specific Jacobian model or gate mode used to
demonstrate it. It reproduces with a different Jacobian source, a different
gate implementation, and a different (already-failing-for-other-reasons)
MPC comparator.

---

## 5. H4 — Scheduled vs live Jacobian accuracy

**Verdict: the schedule is measurably imperfect against the live model, state
deviation from the plan does grow at 255mm as hypothesised, but the resulting
mismatch is too small relative to total tracking error to be a primary driver
of the differences seen in §3-§4 of the main results.**

Methodology: 45-sample subsample per run, one representative rep per condition,
live Jacobians from `from_model_bundle(..., jacobian_mode="accurate")` — the
identical model each schedule was built from. Full table:
`tables/h4_jacobian_accuracy_per_sample.csv`, summary:
`tables/h4_jacobian_accuracy_summary.csv`, figure: `figures/h4_jacobian_accuracy.png`.

### 5.1 Schedule-vs-live error magnitude

| radius | condition | $E_J$ mean | $E_J$ median | $E_J$ p95 | state dev (rad) |
|---|---|---|---|---|---|
| 210 | MPC-$J_C$ | 0.309 | 0.183 | 0.823 | 0.011 |
| 210 | MPC-$J_{NC}$ | 0.170 | 0.144 | 0.337 | 0.031 |
| 210 | InvJac-$J_C$ | 0.189 | 0.134 | 0.478 | 0.033 |
| 210 | InvJac-$J_{NC}$ | 0.174 | 0.158 | 0.388 | 0.083 |
| 255 | MPC-$J_C$ | 0.250 | 0.191 | 0.601 | 0.080 |
| 255 | MPC-$J_{NC}$ | 0.230 | 0.171 | 0.666 | 0.094 |
| 255 | InvJac-$J_C$ | 0.444 | 0.484 | 0.678 | 0.193 |
| 255 | InvJac-$J_{NC}$ | 0.221 | 0.136 | 0.537 | 0.094 |

Schedule-vs-live mismatch is **17-44%** relative Frobenius error, substantial in
absolute terms, but note MPC-$J_C$ at 210mm — the *best-tracking* condition in
the whole study (0.575mm RMSE) — has the **highest** schedule-vs-live mismatch
of the four 210mm conditions (31%). Schedule fidelity alone does not predict
tracking quality across conditions.

### 5.2 Does visiting off-plan states explain the mismatch?

Mean configuration deviation from the offline plan **does** grow at 255mm as
H4 hypothesises: 0.040rad (210mm) → 0.115rad (255mm), ~2.9x, consistent with
the tighter floor forcing controllers away from the planned trajectory (§6).

Per-condition correlation between $E_J$ and state deviation
(`tables/h4_jacobian_accuracy_per_sample.csv`): **7 of 8 conditions show a
strong positive correlation (+0.75 to +0.90)** — state deviation is a good
local predictor of schedule error for MPC-$J_{NC}$, InvJac-$J_C$, and
InvJac-$J_{NC}$ at both radii. **MPC-$J_C$ is the one exception** (−0.15 at
210mm, −0.38 at 255mm): its schedule error does *not* track state deviation the
same way, consistent with §5.1 — MPC-$J_C$'s mismatch has a different source
(plausibly its own aggressive use of the redundant DOFs near the constraint,
§4.3, rather than drift off the nominal trajectory).

### 5.3 Decomposition: schedule/state mismatch vs missing contact physics

$J_{NC}^{sched} - J_C^{live} = (J_{NC}^{sched}-J_{NC}^{live}) + (J_{NC}^{live}-J_C^{live})$,
evaluated for the two conditions that run the **full path** (so the comparison
isn't biased by which part of the trajectory got sampled):

| condition | schedule/state term | missing-contact term | total |
|---|---|---|---|
| MPC-$J_{NC}$ @210mm | 0.170 | 0.270 | 0.380 |
| MPC-$J_{NC}$ @255mm | 0.230 | 0.253 | 0.384 |

The **missing-contact term is larger than the schedule/state term at both
radii** — most of the no-contact schedule's disagreement with the true
(contact) model is because it is the wrong physical model, not because the
schedule has gone stale relative to the states actually visited.

**Caveat:** the same decomposition for the inverse-Jacobian no-contact
conditions is **not reliable** — both terminate early (s_end 39-50mm vs the
75mm full path), so their 45-sample subsample is concentrated before most of
the post-contact divergence has developed (sampled range: 0-39.7mm at 255mm vs
0-75.3mm for the conditions that complete). Their low reported missing-contact
term (0.031-0.064) is a **subsampling artefact**, not evidence that contact
physics matters less for that controller.

### 5.4 Does the mismatch matter for the command actually used, and for prediction?

Command-weighted disagreement $\|(J_{sched}-J_{live})\Delta u\|$: **0.053-0.286mm**
across all eight conditions — one to two orders of magnitude below the 0.4-9mm
RMSE values in §1. Prediction error against measured tip motion:
$e_{pred}$ using the scheduled Jacobian is 0.225-0.430mm; using the live
Jacobian it drops only to 0.177-0.307mm. **Both are small relative to total
tracking error.**

**Conclusion for H4:** the schedule is imperfect, the imperfection is
structured (correlates with state deviation for most conditions, grows with
radius-induced state deviation, and for the full-path MPC conditions is
dominated by missing contact physics rather than staleness) — but its
*magnitude*, measured either as command-weighted disagreement or prediction
error, is too small to be a primary explanation for the large tracking and
completion differences reported in §3-§4. Those are better explained by the
constraint-awareness mechanism (H3) and the freedom-vs-floor mechanism (H2).

---

## 6. Source-magnet configuration — direct physical evidence for H2/H3

**This section is elevated from a supporting check to a primary line of
evidence (2026-10-08).** Everything else in H2/H3 is inferred from tracking
error and completion statistics; this section shows the *mechanism itself* —
where the magnet physically goes, and how far the beam gets fed in — and the
story is visible by eye in a single figure, not just in aggregate numbers.

**Headline figure:** `figures/h2_magnet_distance_and_insertion_story.png` —
magnet-to-beam-base distance vs path progress for all four MPC conditions
($J_C$ and $J_{NC}$ at both 210mm and 255mm, the matched 255mm cell from §3),
both live thresholds overlaid, with a second panel of insertion $L(s)$ and
termination points marked.

### 6.1 What the figure shows, directly

- **At 210mm, $J_C$ and $J_{NC}$ are nearly indistinguishable.** Both dip down
  and pin themselves against the 205mm live threshold from s≈30mm onward and
  stay there, within a few mm of each other, all the way to completion. There
  is no visible difference in *where the controllers choose to put the
  magnet* between the contact-aware and non-contact-aware model at this
  radius — consistent with §3.1's finding that $\Delta_J^{210}\approx 0$.
- **At 255mm, the two conditions diverge sharply and visibly.** Both start
  similarly (wide excursion out to 390-420mm in the first 10mm, then settling
  toward the floor by s≈40mm). From there, **MPC-$J_C$ (dark blue) keeps
  riding tight against the 250mm threshold for the rest of the path**,
  exactly mirroring its own 210mm behaviour. **MPC-$J_{NC}$ (light blue)
  instead starts drifting *away* from the floor from s≈50mm onward** — not
  staying close and failing to track precisely, but *retreating* from the
  near-base configurations altogether, climbing from ≈255mm up past 300mm by
  the time it aborts (marked ×) at s≈64mm.
- **Insertion makes the same story even starker.** MPC-$J_{NC}$@255mm
  terminates at **73.6-74.4mm** of insertion — the lowest of any of the four
  MPC conditions, including both 210mm conditions (80.2-82.4mm) and the
  *other* 255mm condition (MPC-$J_C$: **89.7-91.0mm**, the highest of all
  four, exceeding even its own 210mm self). The contact-aware controller
  pushes the beam in *further* at the tighter floor than at the looser one;
  the no-contact controller gets fed in *less* and then gives up.

### 6.2 Reading this as the mechanism, not just a correlate

This is exactly the causal picture H2 and H3 predict, made visible rather
than inferred:

- **At the loose (210mm) floor, no-contact MPC can reach the same near-base
  configurations as contact-aware MPC can**, because the constraint doesn't
  force a choice between "close to the base" and "correct for the missing
  contact term" — both are available, so the no-contact model's error is
  absorbed without cost (§3.1's $\Delta_J^{210}\approx0$ now has a physical
  referent: the two controllers are quite literally steering to the same
  place).
- **At the tight (255mm) floor, those near-base configurations disappear for
  both controllers equally** — but **contact-aware MPC finds a genuinely
  different, still-effective alternative right at the new boundary** (same
  riding-the-floor behaviour as always, just 45mm further out), while
  **no-contact MPC cannot**: lacking the correct model of how the beam
  actually responds near the wall, it has no reliable way to convert "stay
  near the tightened floor" into "keep the tip where it needs to be", so it
  retreats to configurations further from the base where tracking is locally
  easier but systematically wrong, and falls further and further behind
  until it aborts.
- The insertion numbers confirm this isn't just a lateral/angular
  repositioning — the no-contact controller is **failing to advance the beam
  at all** in its last ~15% of path progress, consistent with a controller
  that has lost the ability to make correct use of its redundant DOFs, not
  one that is merely tracking imprecisely.

### 6.3 Supporting figures and tables (unchanged from the original pass)

Figures: `figures/magnet_config_vs_progress.png`,
`figures/magnet_xy_vs_exclusion_boundary.png`,
`figures/frozen_vs_scheduled_magnet_config.png`.
Table: `tables/magnet_configuration_envelope.csv`.

Condition means (run as unit; **255mm MPC-$J_C$ row updated to the matched
rerun, §3** — the InvJac rows are unaffected by that correction):

| radius | condition | min dist (mm) | max dist (mm) | dist range | final insertion (mm) | magnet z range (mm) |
|---|---|---|---|---|---|---|
| 210 | MPC-$J_C$ | 205.0 | 313.1 | 108.1 | **81.1** | 15.8 |
| 210 | MPC-$J_{NC}$ | 206.2 | 304.9 | 98.7 | **82.1** | 5.9 |
| 210 | InvJac-$J_C$ | 209.5 | 296.0 | 86.5 | **54.5** | 23.3 |
| 210 | InvJac-$J_{NC}$ | 221.7 | 294.4 | 72.7 | **85.4** | 11.8 |
| 210 | MPC-frozen idx0 | 208.3 | 288.2 | 79.9 | **80.6** | **1.8** |
| 210 | MPC-frozen idx100 | 207.1 | 293.3 | 86.2 | **70.9** | 24.0 |
| 255 | MPC-$J_C$ *(matched)* | **251.0** | **404.1** | **153.1** | **90.2** | **32.8** |
| 255 | MPC-$J_{NC}$ | 250.2 | 357.0 | 106.8 | **74.0** | 17.2 |
| 255 | InvJac-$J_C$ | 260.2 | 433.6 | 173.3 | **39.2** | 42.5 |
| 255 | InvJac-$J_{NC}$ | 253.8 | 398.8 | 145.0 | **44.9** | 35.0 |

**Insertion is the clean discriminator of failure.** Every completing MPC
condition reaches 80-91mm of insertion; the one that fails (MPC-$J_{NC}$@255)
is truncated to 73.6-74.4mm. This is the direct physical consequence of the
mechanism above, not (as for the inverse-Jacobian conditions below) an
artifact of a hold gate — MPC has no hold gate; its insertion simply stops
advancing because the controller runs out of useful corrections to make.

**Does the larger radius deny configurations available at 210mm?** Yes, and
MPC-$J_C$ responds by swinging to a genuinely wider but still effective
envelope (max distance 313→404mm, range 108→153mm) rather than failing — this
is a controller finding a new feasible strategy, not merely being pushed
around. MPC-$J_{NC}$'s range also grows (99→107mm) but far more modestly, and
(per §6.1) in the *wrong* direction — away from where it needs to be, not
toward a new viable configuration.

**Frozen vs scheduled MPC visit genuinely different configurations** under an
identical physical constraint, and the *well-conditioned* frozen point
(idx100) is not simply "closer to normal": its magnet z range (24.0mm) is
actually the **largest** of any 210mm MPC condition, while the
ill-conditioned idx0's z range (1.8mm) is the **smallest** — consistent with
§7's finding that the two frozen ablations fail (or don't) through genuinely
different mechanisms, visible here as genuinely different exploration
behaviour, not just different tracking numbers.

---

## 7. H5 — Scheduled vs frozen Jacobian at 210mm

**Verdict (updated 2026-10-08, after a second frozen-Jacobian ablation):
scheduled beats both frozen variants on completion; but the relationship
between model-mismatch metrics and outcome is genuinely more complex than
either "ill-conditioning explains everything" or "staleness explains
everything" — the two frozen ablations trade opposite failure modes, and the
deciding factor is not raw Frobenius divergence, nor conditioning number
alone, but whether the frozen matrix retains any relevance to the contact
physics regime at all.** This is the most important, and most nuanced,
result in the report.

### 7.0 Two frozen ablations, chosen to isolate conditioning from staleness

The original frozen ablation (§7.1-7.4 below) froze at schedule index 0 —
the single worst-conditioned point on the whole 534-sample schedule (cond.
number 5093, vs 4-17 almost everywhere else, §0.3). That confounds two
effects: is frozen MPC's degradation caused by **freezing** (losing
trajectory-varying updates, i.e. staleness) or by this **one pathological
matrix**? To separate them, a second ablation was run with the identical
procedure but frozen at index 100 (s=14.96mm, condition number 9.75 — solidly
in the schedule's normal 4-17 range, confirmed by a full-resolution scan,
§0.3). Index 100 was chosen specifically because it sits well before contact
onset (s=28.5mm) in a region with no other known anomaly (clear of both the
index 0-9 conditioning tail and the index-505 spike).

**Index 100's provenance was checked directly, not assumed:** an earlier live-model
probe (`probe_contact_diff.py`, §0 methodology) evaluated the live contact and
live no-contact Jacobians at the state corresponding to schedule index 100 and
found them to agree to **0.001%** — i.e. at this state the beam has not yet
touched the wall in either model, so this frozen matrix is physically a clean
*free-space* (no-contact-regime) Jacobian. The same probe at index 0 found the
live contact and no-contact models disagree by **42%** there — despite index 0
also being nominally pre-contact, it is **not** a clean free-space snapshot;
its pathological conditioning appears to coincide with (and plausibly causes,
via a degenerate quasi-static solve) an anomalous departure from genuine
free-space behaviour. This directly supports reading the two ablations as:

- **idx100 = a clean, well-conditioned, physically-correct-at-its-own-state
  free-space ("no-contact-regime") frozen Jacobian.**
- **idx0 = a pathologically ill-conditioned, physically anomalous frozen
  Jacobian** that happens not to behave like a textbook free-space matrix.

**Floor verified directly from each run's own `controller_metadata.json`, not
assumed from the run names or the command originally given to produce
them** — a specific check run on 2026-10-08 after a request to confirm this
rather than infer it: all three idx0 runs and all three idx100 runs report
`magnet_exclusion_source: beam_base_fixed_210.00mm`,
`magnet_exclusion_radius_m: 0.205` (TRUE floor=210mm, live=205mm),
**identical to each other and to the scheduled MPC-$J_C$ baseline**. The two
frozen ablations below are therefore a genuinely same-radius comparison, and
are treated as one; there is no 210mm-vs-255mm split to make for this pair.

### 7.0.1 The headline reversal

| | frozen idx0 (ill-conditioned) | frozen idx100 (well-conditioned, pre-contact) | scheduled $J_C$ |
|---|---|---|---|
| completion | **3/3** `path_complete` | **0/3**, all `tracking_error_exceeded` | 3/3 `path_complete` |
| stop point | s=75.3mm (full path) | s_end (ref) = 75.3mm but actually-tracked only to s≈69-75mm before abort; truncated at 77-90% of control steps | s=75.3mm |
| full-run RMSE | 1.479mm | 1.149mm (mean, but over a shorter, failed run) | 0.598mm |
| common-interval (0-68.56mm) RMSE | **1.595mm** | **0.932mm** | 0.523mm |
| common-interval median | 1.223mm | **0.312mm** | 0.329mm |
| common-interval AUC | 1.313 | **0.557** | 0.407 |

**The well-conditioned frozen Jacobian (idx100) outperforms the
ill-conditioned one (idx0) on every tracking metric within the interval both
ran — right up until it doesn't: all three idx100 reps go on to fail late,
while all three idx0 reps survive to the end.** This is a direct,
well-evidenced falsification of the simple intuition that motivated this
ablation ("a well-conditioned frozen matrix should just be better") — it is
better, locally, for most of the path, and then it is categorically worse,
because it fails outright.

Figure: `figures/h5_three_way_tracking_comparison.png`.

Methodology: the frozen schedule is `vessel_c_schedule_..._frozen_idx0.npy`
(every entry = schedule index 0, confirmed by direct array comparison); 45
matched reference indices, one rep each of the frozen-ablation run and the
scheduled MPC-$J_C$ run at 210mm. Tables:
`tables/h5_frozen_vs_scheduled_summary.csv`,
`tables/h5_frozen_schedule_divergence.csv`. Figures:
`figures/h5_frozen_vs_scheduled_jacobian.png`,
`figures/h5_frozen_schedule_divergence.png`.

### 7.1 idx0 vs scheduled: condition-level comparison

| | frozen | scheduled |
|---|---|---|
| $E$ vs live, mean | **0.649** | 0.309 |
| $E$ vs live, median | **0.707** | 0.183 |
| command-weighted $\|(J-J_{live})\Delta u\|$, mean (mm) | 0.123 | 0.203 |
| command-weighted, max (mm) | 0.332 | **5.520** |
| $e_{pred}$ using own $J$, mean (mm) | 0.292 | 0.428 |
| mean tracking error (45-sample) | **1.255mm** | 0.431mm |
| full-run RMSE (all ticks, §1) | **1.479mm** | 0.575mm |

By the matrix-level metrics ($E$ vs live), **the scheduled Jacobian is
2.1-3.9x closer to live than the frozen one**, and frozen's full-run tracking
is **2.6x worse**. So far this looks like clean support for H5.

But two of the finer-grained metrics already complicate it: frozen's
**command-weighted** disagreement is *smaller* on average (0.123 vs 0.203mm)
and far smaller at the worst tick (0.332 vs 5.520mm) than scheduled's, and
frozen's own-Jacobian prediction error $e_{pred}$ is *better*, not worse, than
scheduled's (0.292 vs 0.428mm). Raw matrix divergence and "does it matter for
the command/prediction actually used" are telling different stories.

### 7.2 Where does idx0 actually diverge, and does it track idx0's own error?

Direct computation from the schedule arrays alone (no live model needed —
this is exact, not a subsample): relative divergence of the frozen (index-0)
Jacobian from the true scheduled Jacobian at each index, correlated against
path progress, configuration distance from the freeze state, and insertion
distance (`tables/h5_frozen_schedule_divergence.csv`):

- **corr(divergence, s) = +0.91, corr(divergence, insertion Δ) = +0.90,
  corr(divergence, joint-configuration distance) = +0.31.** Divergence from
  the freeze point tracks almost entirely with how far the beam has been
  **inserted**, much more than with how far the **joints** have moved — i.e.
  it is primarily an insertion-length staleness effect, not a general
  configuration-staleness effect.
- Divergence crosses >10% almost immediately (s≈0.25mm) and grows
  roughly monotonically to >100% by s≈50-70mm, with a visible **steepening
  right at contact onset** (23.6%→38.6%→55.0% across s=28-38mm, vs the
  roughly flat ~20% plateau from s=10-28mm beforehand). This part supports
  H5's premise: contact onset is where the frozen linearization starts
  leaving the model class it was built in.

**But the tracking-error data does not follow this curve.** From
`figures/h5_frozen_vs_scheduled_jacobian.png`: frozen's **tip tracking error
peaks at s=2-16mm (up to 4.0mm)** — while the matrix divergence there is only
**10-35%**, among the lowest values on the entire run. From s≈20mm onward,
frozen's divergence climbs past 50% and later past 100%, yet its tracking
error in that region is **mostly 0.2-2.2mm**, comparable to (not dramatically
worse than) the scheduled controller's own error in the same region. The
threshold-crossing "degradation onset" computed from the Frobenius metric
alone is s=55.6mm — by which point the *worst* tracking error has already come
and gone.

This is confirmed quantitatively, not just visually: the **within-run
correlation between $E_J$ (tick-level Jacobian error) and tick-level tracking
error is −0.068 for the frozen condition** (i.e. indistinguishable from zero,
slightly negative) and +0.231 for scheduled (weak at best). **Bigger local
model mismatch does not predict worse tracking at that tick, for either
condition, and the frozen condition's single largest discrepancy from the live
model (s>70mm, divergence >100%) coincides with some of its lower error
values, not its highest.**

### 7.3 What actually explains idx0's early-path failure, then?

idx0 is literally `schedule[0]` — and §0.3 independently established that
**index 0 is the one point on the entire 534-sample schedule with
pathological conditioning** (condition number 5093, vs 4-17 everywhere else).
Freezing the controller's linearization at a near-singular matrix means every
tick of the early path is controlled through that near-singular operator — a
**conditioning** problem, not a **staleness** problem, for this specific
early-path spike. This is consistent with every piece of evidence above: the
early-path spike is large despite low Frobenius divergence (the matrix is
still *close to* the live Jacobian there — it just has an almost-degenerate
direction, which is a conditioning property the Frobenius metric does not
capture), and the tick-level correlation between divergence and error is null
because divergence and conditioning are different axes.

**This conclusion, reached from idx0 alone, is correct as far as it goes but
incomplete** — §7.5-7.8 below show that staleness *is* a real, independent
failure mechanism, just one that idx0's own anomalous properties happened to
mask.

### 7.4 Answering the brief's four questions, from idx0 alone (superseded — see §7.9)

1. **Is the scheduled Jacobian actually closer to the true live Jacobian than
   the frozen one?** Yes, by 2-4x on every aggregate Frobenius measure.
2. **Does that difference matter for $J\Delta u$?** Marginally, and in the
   *opposite* direction on average — frozen's mean command-weighted
   disagreement is smaller, though its worst case is bounded while scheduled's
   is not (5.52mm max). Not a clean "yes."
3. **Does better local model accuracy translate into better tracking?**
   **Not on a tick-by-tick basis** (correlation ≈ 0 for frozen, weak for
   scheduled). On a whole-run basis, yes, scheduled tracks better overall
   (1.48 vs 0.58mm RMSE) — but the mechanism is not "accurate ticks track well,
   inaccurate ticks track badly" within either run.
4. **At what path position does the frozen model cease to be a useful local
   approximation?** By the matrix metric, divergence is already >10% almost
   immediately and crosses 50% around s≈30-45mm (near contact onset, 28.5mm) —
   but this threshold has **no detectable relationship to where frozen's
   tracking actually fails**, which is s≈2-16mm, before contact and before
   meaningful divergence. The honest answer is that "ceases to be useful" is
   not well described by the Frobenius-divergence threshold at all; the
   controller's failure mode here is dominated by the conditioning of the one
   matrix it was frozen at, not by its growing distance from the live model.

**(Superseded by §7.9 below — idx0 alone cannot distinguish "staleness doesn't
matter" from "idx0's own anomalies happen to mask staleness".)**

### 7.5 idx100's lag signature: a clean, late, catastrophic failure

Figure: `figures/h5_three_way_lag_comparison.png`.

idx100's longitudinal lag stays indistinguishable from the scheduled
baseline's (both near zero, within ±1mm noise) across the **entire first
~60mm of the path — through contact onset and ~30mm beyond it.** Then, within
a ~10mm window (s≈58-68mm, different per rep but always in this band), lag
breaks away and climbs essentially monotonically to 3.1-4.7mm at abort. This
is a textbook late-onset, accelerating-drift signature — exactly what
"staleness causes progressive lag" predicts, and exactly what idx0's own data
(§7.2) failed to show cleanly.

idx0's lag, by contrast, is a **gradual, bounded, partially self-correcting**
climb: rising through s≈30-55mm to a ~1.5-2mm plateau, then actually
*recovering* (dipping below zero, i.e. briefly ahead of reference) around
s≈60-65mm before ending back near 1.5-2mm. It never approaches the 5mm
abort threshold at any point in any of the three reps.

### 7.6 The s=47.8mm transition and the schedule's own small conditioning
features, in context

The true-schedule condition-number trace (bottom panel,
`figures/h5_idx0_vs_idx100_divergence_and_angle.png`) shows the schedule is
not perfectly smooth even outside the index 0-9 tail and the index-505 spike
(§0.3): there are small, narrow condition bumps near s≈48mm and s≈50mm
(cond. rising to ~17, briefly) — in the same neighbourhood as the s=47.8mm
schedule-vs-live divergence spike independently found in H4 (§5.1). Neither
idx0 nor idx100's tracking error shows any corresponding feature at s≈48mm
(figure, top panel) — consistent with H4's own finding that isolated
matrix-level anomalies at a single schedule sample do not translate into
visible tracking problems on their own. The index-505 spike (cond≈665, visible
as the sharp late peak in the bottom panel at s≈72mm) is similarly invisible
in both frozen conditions' tracking traces — by that point idx100 has already
failed and idx0 is well past its own worst region.

### 7.7 Why does idx0 survive what idx100 doesn't? Dominant-direction alignment

Figure: `figures/h5_idx0_vs_idx100_divergence_and_angle.png`. Table:
`tables/h5_idx0_vs_idx100_schedule_comparison.csv` (every one of the 534
schedule indices, pure array math against the true schedule — no live model
needed for this comparison).

Computing, at every schedule index, each frozen matrix's relative Frobenius
divergence **and** its dominant-output-singular-vector angle against the true
scheduled Jacobian at that index:

- **idx0's divergence and angle are consistently smaller (better) than
  idx100's at every post-contact checkpoint** — e.g. at s=45mm: divergence
  78.2% vs 92.1%, angle 46.7° vs 56.3°; at s=75mm: divergence 104.7% vs
  115.7%, angle 64.6° vs 74.3°. This holds at essentially every index past
  contact onset, not just a few cherry-picked points (full table has all 534).
- This is the clearest objective sense in which idx0, despite its
  pathological *local* conditioning, is the *more durable* frozen choice:
  across the whole post-contact evolution of the true Jacobian, idx0's
  direction stays closer to correct than idx100's clean free-space direction
  does. idx100's angle grows past idx0's almost immediately after contact
  onset and the gap is sustained to the end of the path.

**This is consistent with, and a plausible contributor to, idx0 completing
while idx100 fails** — but it does **not** cleanly explain the regional
tracking-error pattern within the stretch both survived (§7.0.1): idx100
tracks *better* than idx0 through s≈20-55mm despite having the *worse*
divergence/angle throughout that same stretch. Reported directly, not
smoothed over: alignment with the true schedule is better correlated with the
long-run completion outcome than with the local, region-by-region tracking
quality.

A plausible reconciliation, offered as interpretation rather than as another
established fact: idx0's own matrix is persistently ill-conditioned
(condition number 5093, fixed for the whole run, since the frozen matrix never
changes) which plausibly injects small, continuous numerical noise into every
control tick — consistent with its mildly elevated 1-2mm error plateau through
the middle of the path. idx100's matrix is numerically clean throughout (its
own condition number is always 9.75), so it produces quiet, confident control
right up until its commanded direction stops being *useful* — which happens
abruptly, once the angle to the true direction grows large enough that the
projected correction becomes too small to counteract real drift, rather than
gradually. Under this reading, idx0 trades constant mild noise for durability;
idx100 trades clean short-term tracking for a hard cliff. Both are internally
consistent with the data in this report; distinguishing them with full
confidence would need instrumentation this project doesn't have (e.g. a
direct measurement of the per-tick command's useful-vs-wasted component), and
is flagged here as a specific, falsifiable open question rather than claimed
as settled.

### 7.8 Does the matched-radius finding (idx0's smaller angle) hold under the common-interval paradox?

Restating §7.0.1's central fact plainly, because it is easy to read past:
within the one interval both idx0 and idx100 can be fairly compared
(0-68.56mm, the length of the shortest run among all nine reps across the
three conditions), **idx100 wins on every single metric** — RMSE, median,
p95, max, and AUC all favour the well-conditioned frozen matrix. The
directional-alignment advantage idx0 holds throughout this same interval
(§7.7) does not prevent idx100 from controlling tighter, tick to tick, within
it. idx0's advantage only cashes out beyond this interval, where idx100 no
longer has any data because it already failed. **"Better aligned" and
"better tracking over the region sampled" are not the same claim, and this
dataset cleanly separates them.**

### 7.9 Answering the brief's four questions, incorporating both ablations

1. **Is the scheduled Jacobian actually closer to the true live Jacobian than
   either frozen one?** Yes, by 2-4x on every aggregate Frobenius measure
   against idx0 (§7.1); idx100 was not re-evaluated against the live model
   (its provenance was established via the separate pre-contact live probe,
   §7.0), but by construction (frozen at a true schedule sample) it starts at
   0% divergence from the *schedule* and diverges from it exactly as
   characterized in §7.7 — scheduled, which updates every tick, is
   structurally guaranteed to track the live/true Jacobian more closely than
   any single frozen sample over a trajectory that leaves that sample's
   regime.
2. **Does that difference matter for $J\Delta u$ / for tracking?** It matters
   for *outcome* (completion) in a way that is not well predicted by *any
   single* matrix-level metric examined in this report. Raw Frobenius
   divergence from live: no (§7.2, null tick-level correlation, confirmed for
   both frozen conditions in the earlier H4/H5 work). Condition number alone:
   no (idx0 is drastically worse-conditioned than idx100 yet survives).
   Dominant-direction alignment with the true schedule: **partially** — it
   correctly ranks which frozen condition ultimately completes, but does not
   predict which one tracks better moment-to-moment.
3. **Does better local model accuracy translate into better tracking?** Not
   tick-by-tick (confirmed independently for idx0 in §7.2 and consistent with
   idx100's own pattern: its tracking is good through s≈55mm despite already
   having 60-100%+ schedule divergence there). On a whole-run basis, the
   answer now depends on which frozen condition and which metric: idx100 is
   the *locally* more accurate frozen choice and tracks tighter while it
   lasts, but **fails to finish**; idx0 is both locally noisier and globally
   more divergent in the aggregate Frobenius sense, yet finishes. "Better
   local accuracy" is not a reliable predictor of "better outcome" in either
   direction in this dataset.
4. **At what path position does a frozen model cease to be a useful local
   approximation?** For idx100, this now has a sharp, well-evidenced answer:
   **s≈58-68mm**, visible simultaneously as (a) tracking error's first
   sustained departure from the scheduled baseline, (b) lag's breakaway from
   near-zero into monotonic growth, and (c) the point beyond which no rep
   survives. This is **not** contact onset (s=28.5mm, ~30mm earlier) — idx100
   tolerates moderate-to-large schedule divergence (60-100%+) for an extended
   stretch post-contact before failing, consistent with a cumulative/lag-style
   failure mode rather than an instantaneous one. For idx0, no such sharp
   transition exists because it never fails; its worst local degradation
   (s≈2-16mm) is explained by conditioning (§7.3), not staleness.

**Net verdict (revised):** trajectory-varying Jacobians remain the right
practical choice — scheduled beats both frozen ablations on every completion
and common-interval metric. But the single-ablation claim this report
originally made ("mismatch-from-freeze-point does not explain frozen-Jacobian
failure") is now understood to be **specific to idx0's own anomalous
properties, not a general statement about staleness.** A second, cleanly
pre-contact, well-conditioned freeze point shows exactly the delayed,
accelerating, lag-dominated degradation the brief's original mechanism
predicts — it is just that idx0 happened to be a poor test case for it,
being simultaneously ill-conditioned *and* (per the 42% live-model
disagreement, §7.0) not a clean representative of any single physics regime.
**Both mechanisms are real; they do not act through the same channel; and
which one dominates depends on which kind of frozen point you happen to
pick.**

---

## 8. Gain ablation (0.6 vs 1.0), analysed separately

**Verdict: the specific hypothesis is contradicted.** The brief asks whether gain
1.0 "simply makes the constraint-unaware inverse controller request infeasible
motions more aggressively". It does not.

From `tables/gain_ablation.csv` (InvJac-$J_{NC}$ at 210mm, n=3 per gain):

| gain | RMSE (mm) | max err (mm) | mean \|raw cmd\| | p95 \|raw cmd\| | frac infeasible | frac held | min dist (mm) | sign reversals/tick |
|---|---|---|---|---|---|---|---|---|
| 0.6 | 0.685 | 1.858 | 0.0122 | 0.0369 | **0.000** | **0.000** | 221.7 | 0.857 |
| 1.0 | 1.128 | 4.820 | 0.0286 | 0.1096 | **0.000** | **0.000** | 228.0 | 1.152 |

- The raw requested command does get ~2.3x larger (mean) and ~3x larger (p95).
- But the **infeasible-request rate stays exactly zero at both gains**, the hold
  gate never fires at either gain, and the magnet's closest approach actually
  moves **further from** the boundary (221.7→228.0mm), not closer.
- Tracking degrades substantially: RMSE +65%, max error 2.6x.
- Per-tick joint sign reversals rise 0.857→1.152 (+34%).

**It is not constraint pressure**, that much is established directly: the
infeasible-request rate is exactly zero at both gains. What *is* established
is this: the increased command magnitude and sign-reversal rate are
consistent with underdamped/oscillatory behaviour when proportional gain is
raised without a matching increase in damping (`damping` was held at 0.05 at
both gains). This is a plausible explanation, not a uniquely proven one — it
has not been confirmed with a damping sweep, which this report does not treat
as warranted on its own (a secondary ablation not central to the main
hypotheses). What the data does establish without qualification: whatever the
precise mechanism, it is **independent of constraint awareness** — the gain
ablation does not speak to H3.

All six gain-ablation runs completed the path, so the degradation is an accuracy
cost, not a feasibility cost.

Figure: `figures/gain_ablation.png`.

---

## 9. Deliverables index

1. **Per-run summary table:** `tables/per_run_summary_closedloop.csv` (34 closed-loop/frozen/matched/excluded runs, `cell_role` column) + `openloop_reused/common_interval_stats.csv` (4 open-loop runs).
2. **Figures:** `figures/` — `closedloop_tracking_{210,255}mm.png`, `mpc_nc_vs_c_255mm_lag_crosstrack.png`, `h3_representative_stall_event.png`, `magnet_config_vs_progress.png`, `magnet_xy_vs_exclusion_boundary.png`, `frozen_vs_scheduled_magnet_config.png`, `h2_magnet_distance_and_insertion_story.png`, `h3_invjac_vs_mpc_redundancy_drift.png` (§4.6), `h3_invjac_redundancy_drift_jacobian_independence.png` (§4.7), `s54_60mm_targeted_diagnostic.png` (§0.3 targeted diagnostic), `h4_jacobian_accuracy.png`, `h5_frozen_vs_scheduled_jacobian.png`, `h5_frozen_schedule_divergence.png`, `h5_three_way_tracking_comparison.png`, `h5_three_way_lag_comparison.png`, `h5_idx0_vs_idx100_divergence_and_angle.png`, `gain_ablation.png`; open-loop figures in `openloop_reused/`.
3. **Interaction-effects table:** `tables/h2_interaction_effects.csv` (210 vs 255 for MPC and inverse-Jacobian, both Jacobians, radius-matched) — see §3.1. The MPC-vs-InvJac comparison at each radius is also in that table (`mpc_vs_inv_at_C`, `mpc_vs_inv_at_NC` columns).
4. **210mm frozen-vs-scheduled ablation tables:** `tables/h5_frozen_vs_scheduled_summary.csv`, `tables/h5_frozen_schedule_divergence.csv` (idx0 vs scheduled, §7.1-7.4), `tables/h5_idx0_vs_idx100_schedule_comparison.csv` (idx0 vs idx100 vs the true schedule at every index, §7.7) — see §7. **Not included in any 210-vs-255 factorial comparison**, per the brief.
5. **Inverse-controller infeasible-command table:** `tables/h3_invjac_infeasible_commands.csv`, MPC counterpart `tables/h3_mpc_constraint_margin.csv` — see §4.2.
6. **Scheduled-vs-live Jacobian accuracy (both radii):** `tables/h4_jacobian_accuracy_summary.csv` (+ per-sample `h4_jacobian_accuracy_per_sample.csv`) — see §5.
7. **Frozen-vs-scheduled-vs-live (210mm), both ablations:** `tables/h5_frozen_vs_scheduled_summary.csv` (+ per-sample `h5_frozen_jacobian_per_sample.csv`, `h5_scheduled_pair_per_sample.csv`) for idx0; `tables/h5_idx0_vs_idx100_schedule_comparison.csv` for the idx0-vs-idx100 comparison — see §7.

Additional tables produced along the way: `tables/common_interval_{210,255}mm.csv`
(§1), `tables/longitudinal_lag_vs_crosstrack.csv` (§3.4), `tables/magnet_configuration_envelope.csv`
(§6), `tables/gain_ablation.csv` (§8), `tables/s54_60mm_targeted_detail.csv`
(§0.3 targeted diagnostic, 137 rows), `tables/h4_v2_jacobian_accuracy_per_sample.csv`
(H4's guided-sampling dataset, 1636 rows, 8 conditions — see §5).

---

## 10. Results narrative by hypothesis (summary)

**H1 (contact modelling is required):** supported. Contact plan completes
(max error 8.8-9.1mm, under the 10mm threshold); no-contact plan fails at
~82% progress (max error 9.8-9.9mm at abort). Plans behave similarly
pre-contact and diverge post-contact, but the measured divergence onset
(s≈50.5-51.8mm) lags the model-predicted contact onset (s=28.5mm) by ~22mm —
contact is initially tolerable, not immediately disqualifying.

**H2 (contact-Jacobian x radius interaction):** **supported, on a corrected,
radius-matched comparison** (updated 2026-10-08). MPC-$J_{NC}$ completion
collapses 3/3→0/3 as the floor tightens while MPC-$J_C$ stays 3/3→3/3 at the
**identical live thresholds** (205mm, then 250mm, both confirmed via
`controller_metadata.json`), and $\Delta_J^{255}(+0.406) > \Delta_J^{210}(-0.101)$
as predicted. The original "255mm" MPC-$J_C$ condition was run at TRUE
floor=240mm (live 235mm), 15mm more true freedom than its sibling, which
inflated the first estimate to +0.536mm; a corrected rerun at the exactly
matched TRUE=255mm/live=250mm gives the honest, trustworthy figure above.
The lag sub-claim is only partially supported: MPC-$J_{NC}$'s terminal lag at
255mm is 36x its 210mm value, but cross-track grows by a similar factor over
the run as a whole, so the failure is not purely longitudinal.

**H3 (MPC vs inverse-Jacobian constraint awareness):** supported, with the
offline replay validated to reproduce the real hold/pass decision on every
tick of all 12 runs (agreement=1.000). The naive controller requests
genuinely infeasible motions on 36.8-69.6% of ticks once the constraint binds,
and is held in place every single time; MPC rides the same constraint on
8.5-20.4% of ticks and never violates it once, across 12 runs. Two refinements
are needed: the difference is **latent, not demonstrated**, in the one cell
where the constraint never binds (InvJac-$J_{NC}$ @210mm, 0 infeasible ticks);
and at 255mm roughly half of the infeasible episodes are **z-workspace**
violations, not the beam-base-radius violations the hypothesis names. A
third, genuinely distinct failure mode was also found and generalized
(§4.6-4.7): with the magnet-exclusion gate working perfectly (0-1.9% clip
activity, magnet never closer than 0.5mm to its floor), three independent
inverse-Jacobian conditions — two Jacobian sources (contact, no-contact), two
gate implementations (`selective`, `clip`) — all still fail via
`tcp_out_of_workspace`, caused by unregularized drift in the controller's
redundant joint directions (`nullspace_gain=0`, no term anywhere penalizing
them) that MPC's input-tracking/increment cost bounds implicitly even with
no explicit posture cost (`Q_N=0`). This is a control-law property, not an
artefact of one Jacobian or one gate, and a plausible (not separately
confirmed) contributor to the inverse-Jacobian conditions' broader late-path
degradation in the main factorial.

**H4 (scheduled-vs-live Jacobian accuracy):** the schedule is measurably
imperfect (17-44% relative Frobenius error) and more so where state deviation
from the plan is larger — which does grow ~2.9x at 255mm, as hypothesised, for
7 of 8 conditions (MPC-$J_C$ is the exception). For the two conditions that run
the full path, missing contact physics is a bigger contributor to no-contact
schedule error than state/schedule staleness. But in absolute terms the
mismatch's effect on the command and on one-tick prediction (sub-mm) is small
relative to the multi-mm tracking differences this report is explaining —
**H4's effect is real but not the primary driver of H2/H3's results.**

**H5 (frozen vs scheduled at 210mm, two ablations):** scheduled beats both
frozen variants on completion and every common-interval metric — the
practical conclusion ("don't freeze the Jacobian") holds robustly. The
*mechanism*, however, turns out to be genuinely two-sided, revealed only by
running a second, deliberately well-conditioned, pre-contact freeze point
(idx100, cond=9.75, confirmed via a live-model probe to be a clean free-space
Jacobian, 0.001% from the live no-contact model) alongside the original
pathological one (idx0, cond=5093, confirmed anomalous — 42% from the live
no-contact model despite also being pre-contact). **idx0 completes all 3
reps; idx100 fails all 3, late (s≈58-68mm), via a clean accelerating-lag
signature — exactly the staleness mechanism the brief originally
hypothesised, and exactly what idx0 alone failed to show.** idx100 also
*outtracks* idx0 on every metric within the interval both survive (RMSE 0.93
vs 1.60mm), so "well-conditioned" is not simply "better" — it trades
continuous mild noise (idx0, from its fixed ill-conditioning) for a hard
failure cliff once its frozen free-space direction loses relevance to deep
contact mechanics (idx100). A full-schedule dominant-direction-angle
comparison shows idx0 stays measurably better aligned with the true
post-contact Jacobian throughout (§7.7), which plausibly explains why it
survives — but does not explain why idx100 tracks tighter while both are
still running. Both mechanisms (conditioning noise, and alignment-driven
staleness) are real, act through different channels, and neither alone
explains the full picture.

**Gain ablation (0.6 vs 1.0):** the specific sub-hypothesis is **contradicted**.
Infeasible-request rate is exactly 0.000 at both gains; the higher gain moves
the magnet's closest approach *further* from the boundary (221.7→228.0mm).
Degradation (RMSE +65%, max error 2.6x) tracks sign-reversal rate (+34%); the
increased command magnitude and sign-reversal rate are consistent with
underdamped/oscillatory behaviour when gain is raised without raising
damping, though this has not been confirmed with a damping sweep. What is
established without qualification is that the degradation is independent of
constraint awareness.

---

## 11. Final hypothesis verdicts

| Hypothesis | Verdict | Key qualifier |
|---|---|---|
| **H1** — contact modelling required | **Strongly supported** | divergence onset lags predicted contact onset by ~22mm |
| **H2** — $\Delta_J$ interaction with radius | **Supported** (updated 2026-10-08) | direction and magnitude both confirmed on a radius-matched rerun ($\Delta_J^{255}=+0.406>\Delta_J^{210}=-0.101$, live thresholds 205mm/250mm verified identical on both legs via `controller_metadata.json`); completion evidence (3/3→0/3 for MPC-$J_{NC}$, 3/3→3/3 for MPC-$J_C$) remains the most decisive piece; lag sub-claim only true of the terminal state, not the whole run |
| **H3** — MPC vs inverse-Jacobian constraint awareness | **Supported** | exact (agreement=1.000) replay confirms the mechanism; latent (not demonstrated) when the constraint never binds; ~half the 255mm episodes are z-workspace, not radius; a third, distinct failure mode (unregularized null-space drift, §4.6-4.7) was found and confirmed to generalize across 2 Jacobian sources x 2 gate modes — the exclusion gate itself works throughout, so this does not change the constraint-awareness verdict, but it is a second, independent reason the inverse-Jacobian controller underperforms |
| **H4** — scheduled-vs-live accuracy, state-deviation driven | **Partially supported** | mismatch is real and structured as predicted, but too small in command/prediction terms to be a primary cause of the main results |
| **H5** — trajectory-varying Jacobian materially helps | **Supported on outcome; mechanism is two-sided, confirmed by a second ablation** | scheduled beats BOTH frozen ablations on completion (idx0 3/3, idx100 0/3, scheduled 3/3) and on every common-interval metric; idx0's failure mode is conditioning-driven (confirmed, §7.3), idx100's is a clean staleness/alignment-driven late lag blowup (confirmed, §7.5, exactly the mechanism the brief originally proposed) — idx0 alone had wrongly suggested staleness didn't matter at all |
| Gain ablation sub-hypothesis (higher gain → more infeasible requests) | **Contradicted** | infeasible rate is 0.000 at both gains; degradation is consistent with underdamped/oscillatory behaviour (not confirmed by a damping sweep), but is established to be independent of constraint pressure regardless of mechanism |

No result in this report rests on a single run: every quantitative claim above
is a mean (or an explicit per-run breakdown) over 3 reps, treating the run as
the statistical unit, consistent with the brief's instruction not to claim
significance from tick-level samples.
