# Controller comparison — 10x15mm rectangle, bimaterial model, 2026-09-15

## Setup
- Plan: `/home/jack/.claude/jobs/3710eca5/tmp/plans_bimaterial/rectangle_10x15mm_bimaterial/time_parameterized_configuration_path`
  (10mm x 15mm rectangle, replanned with the bimaterial Cosserat beam model
  built this session, 264 time-parameterized samples).
- Beam length 25.0mm at plan start, same day/calibration state for every run below.
- `state_tracking_weight = 0.001` and `--horizon 15` applied throughout
  (confirmed via each run's `config.json`).
- **n=3 for every controller condition; n=1 for open loop** — same caveat as
  the triangle8mm doc: open loop has no std and is not statistically
  comparable to the n=3 rows.

## Root-cause status on this shape — important asymmetry vs the triangle
Unlike the 8mm triangle (see `close_loop_logs/triangle8mm_bimaterial_analysis_2026-09-15/ROOT_CAUSE.md`),
**there is no rectangle-specific horizon=30-vs-15 before/after comparison**.
Both fixes (`state_tracking_weight=0.001`, discovered on day2's 20mm triangle;
`--horizon 15`, discovered this session on the 8mm triangle) were applied
directly to the rectangle's first clean comparison run. This doc therefore
**validates that both fixes generalize to a third, geometrically distinct
shape** — it does not re-derive either root cause independently on this
shape, and no claim below should be read as rectangle-specific root-causing.

One earlier attempt exists at `close_loop_logs/compare_FF_rectangle10x15mm_bimaterial_2026-09-15/`
(the OLD default `state_tracking_weight`, horizon=30, manually killed mid-run
when the bug was rediscovered — see day2's memory-triggered correction). It
is **not used for any figure or statistic below** because it conflates both
un-fixed variables at once and is incomplete (`inv_rep2` has no `summary.json`,
i.e. was killed mid-rep). For anecdotal context only: its 3 complete reps show
`inv+FF` rms=0.96mm/hold=0.71mm, `mpc_lti+FF` rms=1.12mm/hold=1.09mm,
`mpc_ltv+FF` rms=1.44mm/hold=0.95mm — both MPC variants losing to inv+FF, the
same qualitative symptom the fixes were built to resolve, but this single-rep,
double-confounded data point cannot isolate which fix (or both) is responsible
and is not part of the formal comparison.

## Results (mean ± std; n=3 for controllers, n=1 for open loop)

| Controller | RMS (mm) | Max (mm) | Hold (mm) | Final (mm) |
|---|---|---|---|---|
| Open loop (n=1) | 1.51 | 3.09 | 1.17 | 1.11 |
| Inverse-Jacobian+FF | 1.18 ± 0.11 | 2.77 ± 0.24 | 0.46 ± 0.25 | 0.31 ± 0.23 |
| MPC-LTI+FF (h=15, fixed) | **1.04 ± 0.08** | **2.62 ± 0.21** | 0.31 ± 0.06 | 0.32 ± 0.09 |
| MPC-LTV+FF (h=15, fixed) | 1.08 ± 0.04 | 2.60 ± 0.02 | **0.19 ± 0.07** | **0.11 ± 0.03** |

(Values re-verified from `close_loop_logs/compare_FF_rectangle10x15mm_bimaterial_h15_n3_2026-09-15/summary.json`
and `openloop_rectangle10x15mm_bimaterial_2026-09-15/summary.json`.)

## Headline findings
1. **Both fixed MPC variants beat open loop by 28–31% RMS** (1.04–1.08mm vs
   1.51mm) and beat inverse-Jacobian+FF by 8–12% RMS (1.04–1.08mm vs 1.18mm)
   — the same qualitative ranking as day2 (20mm triangle) and the 8mm
   triangle, on a third, larger, 4-cornered shape. MPC-LTI edges out MPC-LTV
   on RMS (1.04 vs 1.08mm), matching the 8mm triangle's ordering, not day2's
   (where LTV led) — see finding 3.
2. Inv+FF and both MPC variants beat open loop, again confirming the reused
   offline feedforward solution is the dominant contributor to tracking
   accuracy, independent of shape.
3. **Jacobian evolution is the most severe of the three shapes tested this
   session** (`figures/fig5_jacobian_evolution.png`,
   `figures/fig6_jacobian_mismatch_vs_error.png`): condition number ranges
   156–83,011 along the path and the frozen-Jacobian mismatch grows to **158%**
   of ‖J(0)‖ — larger than the 8mm triangle's 94% and closer to day2's 20mm-
   triangle severity. Despite this, MPC-LTV does not clearly outperform
   MPC-LTI here on RMS (1.08 vs 1.04mm) even though LTV should in principle
   benefit most from relinearization when mismatch is large — LTV does win
   decisively on hold error (0.19 vs 0.31mm) and final error (0.11 vs 0.32mm),
   so its advantage shows up in the *terminal* phase, not the *transit* RMS.
   This is a real, measured pattern, not a hypothesis — but the mechanism for
   why it doesn't translate to a transit-RMS win despite the larger mismatch
   is not established here.
4. **Mechanism localization is markedly less clean than on the 8mm triangle**
   (`figures/fig7_mechanism_localization.png`; corners auto-detected at
   reference-sample indices 96, 129, 213 — only 3 of the rectangle's 4
   geometric corners cross the adaptive detection threshold, consistent with
   the closed path's start vertex not producing an internal tangent-angle
   peak, the same effect seen for the triangle's apex-at-start corner). The
   per-bin (inv−mpc) RMS difference alternates sign through most of the path
   (bins showing both +0.35mm and −0.42mm) rather than concentrating in one
   contiguous region the way the 8mm triangle's gap did. Only the last ~3
   bins (samples ≈230–264, approaching the terminal hold) show a
   consistent, sizeable MPC advantage (+0.39 to +0.62mm/bin). Read this as:
   on this shape, the RMS-level localization story from day2/8mm-triangle
   (gap concentrates in one geometric region) does **not** clearly replicate
   — the rectangle's gap looks more distributed across the path, with a real
   but smaller net effect (mean diff +0.08mm/bin) than the localized 8mm
   triangle finding (+0.12mm/bin). This is reported as an honest partial
   replication, not forced into the same narrative.
5. **Controller fingerprint again shows the same direction as day2 and the
   8mm triangle** (`figures/fig8_controller_fingerprint.png`): MPC saturates
   far more (64–75% of ticks vs inv+FF's 26%; notably higher than either the
   8mm triangle's 24–28% or day2's 62–78% — this is the highest saturation
   rate of any shape tested), is jerkier (1.7–2.0 vs 0.9 mrad/tick), and
   routes far more command through the exact Jacobian nullspace (44–54% vs
   0.5%). As before, neither "better anticipation" nor "smoother commands"
   explains MPC's edge — if anything MPC's behavioural profile is *worse* on
   this shape than on the others tested, yet it still wins on RMS/hold/final,
   reinforcing that the win is not behavioural-fingerprint-driven.

## Data
Raw per-tick logs live under `close_loop_logs/compare_FF_rectangle10x15mm_bimaterial_h15_n3_2026-09-15/runs/`
and `close_loop_logs/openloop_rectangle10x15mm_bimaterial_2026-09-15/runs/`.

## Figures (`figures/`)
- **fig1_tip_paths.png** — measured tip path vs reference, representative
  (median-RMS) rep, all 4 conditions.
- **fig2_error_vs_time.png** — tracking error vs time, all 4 conditions overlaid.
- **fig3_summary_bars.png** — RMS/max/hold/final bars, n=1 (open loop, hatched)
  vs n=3 (controllers).
- **fig4_jacobian_svd.png** — frozen beam Jacobian singular-value spectrum at
  plan start (κ≈69,285 — essentially identical to the 8mm triangle's, since
  both plans start from the same 25mm-insertion beam configuration).
- **fig5_jacobian_evolution.png** — local Jacobian singular values / condition
  number vs path progress from the cached LTV relinearization schedule.
- **fig6_jacobian_mismatch_vs_error.png** — ‖J(k)−J(0)‖_F / ‖J(0)‖_F vs path
  progress (peaks at 158%, the largest of the three shapes), overlaid with
  mpc_lti+FF's and mpc_ltv+FF's tracking error.
- **fig7_mechanism_localization.png** — reference speed profile with
  auto-detected corners (3 of 4 found), inv+FF vs best-MPC error vs path
  progress, and a per-progress-bin (inv−mpc) RMS difference bar chart.
- **fig8_controller_fingerprint.png** — step-limit saturation / jerk /
  nullspace-usage fingerprint, inv+FF vs mpc_lti+FF vs mpc_ltv+FF.

No horizon=30-vs-15 diagnostic figure exists for this shape — see "Root-cause
status" above.
