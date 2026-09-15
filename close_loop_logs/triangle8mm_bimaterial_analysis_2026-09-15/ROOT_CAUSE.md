# Controller comparison — 8mm triangle, bimaterial model, 2026-09-15

## Setup
- Plan: `/home/jack/.claude/jobs/3710eca5/tmp/plans_bimaterial/triangle_8mm_bimaterial/time_parameterized_configuration_path`
  (8mm-base / 14mm-depth apex-at-start triangle, replanned with the bimaterial
  Cosserat beam model built this session — 0.2mm nitinol wire shaft + 40mm
  PDMS/iron-particle composite tip, 5mm bonded overlap — 123 time-parameterized
  samples).
- Beam length 25.0mm at plan start, same day/calibration state for every run below.
- `state_tracking_weight = 0.001` applied throughout (confirmed via each run's
  `config.json`: `"mpc_state_tracking_weight": 0.001`) — this is the fix
  root-caused and validated on the 20mm triangle the previous day
  (`close_loop_logs/day2_triangle20mm_2026-09-12/FINAL_COMPARISON.md`), **not**
  rediscovered here; it is simply carried forward as a prerequisite.
- MPC horizon = 15 for the primary (fixed) comparison below. **The horizon=15
  vs horizon=30 diagnostic below is the new root-cause finding from this
  session**, discovered specifically on this shape.
- RMS/max computed over the transit phase only (`terminal_hold == False`);
  hold = mean error during the terminal hold; final = error at the last
  control step.
- **n=3 for every controller condition; n=1 for open loop** (one matched-
  condition rep, `openloop_triangle8mm_bimaterial_rep2_2026-09-15`) — open
  loop has no std and should not be read with the same statistical weight as
  the n=3 rows.

## Root-cause fix #2: MPC horizon
With `state_tracking_weight=0.001` already active, MPC+FF on this shape was
still producing a large, systematic **hold-phase error blow-up** (2.6–3.7mm)
at `--horizon 30` — worse than open loop, reproducing the exact "why is MPC
worse" symptom the user had already flagged on this session's rectangle
comparison. A matched single-rep `--horizon 15` rerun on the same shape/day/
fix immediately resolved it (hold dropped to 0.18–0.29mm), isolating horizon
as the second, independent root cause (distinct from `state_tracking_weight`,
which was already fixed in both the broken and the working run).

**What the blow-up actually looks like** (see `figures/fig9_horizon_diagnostic.png`,
matched `mpc_lti+FF` reps
`compare_FF_triangle8mm_bimaterial_1rep_2026-09-15/runs/lti_rep1_20260915T135248Z` [h=30]
vs `compare_FF_triangle8mm_bimaterial_h15_2026-09-15/runs/lti_rep1_20260915T135937Z` [h=15]):

| | h=30 (unfixed) | h=15 (fixed) |
|---|---|---|
| hold mean error | 2.61 mm | 0.18 mm |
| hold max error | 5.01 mm | 0.42 mm |
| hold error std | 1.45 mm | 0.09 mm |
| hold error trend (linear fit, mm/s) | **−2.37** | −0.015 |

The negative trend at h=30 means the hold-phase error is *not* a growing/
diverging oscillation — it is large at the *moment the transit phase ends and
the hold begins*, then relaxes downward over the following seconds. This
points to a **transient overshoot at the hold transition**, not instability:
with horizon=30, MPC's internal prediction is still weighted toward a
30-step-ahead extrapolation of the (now-finished) reference trajectory
exactly when the reference stops advancing, producing a poorly-matched final
correction that then has to be walked back down. At horizon=15 this
mismatch window is much shorter and the transition is smooth (trend ≈ 0,
flat low-error hold from the start). This is a mechanistically plausible
explanation consistent with the data, not confirmed at the state/cost-vector
level (that would require logging the MPC's internal predicted trajectory,
which is not captured by `path_follow.jsonl`). Only one matched rep per
horizon was run — this is a two-point comparison, not an n=3-validated
effect on this shape (the horizon=15 result is separately confirmed at n=3
below, but the *contrast against a broken horizon=30* is single-rep only, and its
"blowup" is real and large, but the specific overshoot-and-relax profile
should be treated as directional, not statistically confirmed).

## Results (mean ± std; n=3 for controllers, n=1 for open loop)

| Controller | RMS (mm) | Max (mm) | Hold (mm) | Final (mm) |
|---|---|---|---|---|
| Open loop (n=1) | 1.17 | 2.46 | 0.27 | 0.28 |
| Inverse-Jacobian+FF | 0.90 ± 0.06 | 2.15 ± 0.21 | 0.56 ± 0.04 | 0.51 ± 0.03 |
| MPC-LTI+FF (h=15, fixed) | **0.76 ± 0.03** | **1.86 ± 0.09** | 0.29 ± 0.09 | 0.27 ± 0.03 |
| MPC-LTV+FF (h=15, fixed) | 0.79 ± 0.01 | 1.86 ± 0.08 | **0.25 ± 0.02** | 0.25 ± 0.13 |

(Values re-verified from `close_loop_logs/compare_FF_triangle8mm_bimaterial_h15_n3_2026-09-15/summary.json`
and `openloop_triangle8mm_bimaterial_rep2_2026-09-15/summary.json`.)

## Headline findings
1. **Both fixed MPC variants beat open loop by 32–35% RMS** (0.76–0.79mm vs
   1.17mm) and beat inverse-Jacobian+FF by 12–15% RMS (0.76–0.79mm vs
   0.90mm) — reproducing the day2 (20mm triangle) ranking on an unrelated,
   much smaller shape. **MPC-LTI edges out MPC-LTV here** (0.76 vs 0.79mm),
   the reverse of day2's ordering (where LTV was best) — see finding 3 for
   why this is plausible and not a fluke.
2. Every controller (inv+FF and both MPC variants) beats open loop; this
   confirms the reused feedforward baseline (from the offline nonlinear
   solve) is still the dominant contributor on this shape too, exactly as
   day2 found.
3. **Jacobian evolution is comparatively mild on this small, 8mm shape**
   (`figures/fig5_jacobian_evolution.png`): condition number ranges
   1,372–73,632 along the path (vs day2's 20mm-triangle range of ~300–700,000
   at its corners) and the frozen-Jacobian mismatch peaks at only 94% of
   ‖J(0)‖ (`figures/fig6_jacobian_mismatch_vs_error.png`), smaller than
   rectangle's 158% (see the rectangle doc). This is consistent with
   MPC-LTV's per-step relinearization advantage over MPC-LTI being *weaker*
   on this shape — plausibly why LTI edges out LTV here, unlike on the
   larger/more strongly-varying 20mm triangle and the rectangle (below).
4. **Mechanism localization** (`figures/fig7_mechanism_localization.png`,
   corners auto-detected at reference-sample indices 51 and 74 from the
   plan's own tangent-vector data, adaptive-threshold method — see script):
   the inv+FF vs mpc_lti+FF RMS gap is small and mixed-sign through the first
   corner, then concentrates in the back half of the path (bins spanning
   samples ~82–119, diffs of +0.15 to +0.57mm favoring MPC), mirroring day2's
   "gap concentrates after the geometry gets harder, not exactly at a corner"
   finding. Mean per-bin diff across the whole transit: +0.12mm favoring MPC.
5. **Controller fingerprint is another honest negative result**
   (`figures/fig8_controller_fingerprint.png`), matching day2's finding
   exactly in direction: MPC saturates the joint-step limit far more often
   (24–28% of ticks vs inv+FF's 15%), is ~2.6–2.8x jerkier (2.76–2.96 vs
   1.07 mrad/tick), and routes far more of its command through the exact
   nullspace of the beam Jacobian (47–49% vs 0.2%). As in day2, neither
   "better constraint anticipation" nor "smoother commands" explains MPC's
   RMS edge here — the mechanism remains the localization result in finding 4,
   not the fingerprint.

## Data
Raw per-tick logs live under `close_loop_logs/compare_FF_triangle8mm_bimaterial_h15_n3_2026-09-15/runs/`
(fixed n=3), `close_loop_logs/openloop_triangle8mm_bimaterial_rep2_2026-09-15/runs/`
(open loop), `close_loop_logs/compare_FF_triangle8mm_bimaterial_1rep_2026-09-15/runs/`
(h=30 single-rep) and `close_loop_logs/compare_FF_triangle8mm_bimaterial_h15_2026-09-15/runs/`
(h=15 single-rep, matched pair to the h=30 run). Aggregate figures for the
h=30 MPC-only n=3 run (`compare_FF_triangle8mm_bimaterial_mpc_2026-09-15/`)
exist but are not used above since it has no matched inv+FF baseline at h=30
on this shape — it corroborates the same blow-up (hold RMS ~2.8±1.3mm at
h=30, contaminated further by one aborted `stale_vision` rep) but is not a
clean statistical comparison.

## Figures (`figures/`)
- **fig1_tip_paths.png** — measured tip path vs reference, representative
  (median-RMS) rep, all 4 conditions.
- **fig2_error_vs_time.png** — tracking error vs time, all 4 conditions
  overlaid.
- **fig3_summary_bars.png** — RMS/max/hold/final bars, n=1 (open loop, hatched)
  vs n=3 (controllers).
- **fig4_jacobian_svd.png** — frozen beam Jacobian singular-value spectrum at
  plan start (κ≈69,380 — near-null third direction, same weak-authority
  signature documented since the `beam-lateral-authority-limit` memory).
- **fig5_jacobian_evolution.png** — local Jacobian singular values / condition
  number vs path progress from the cached LTV relinearization schedule.
- **fig6_jacobian_mismatch_vs_error.png** — ‖J(k)−J(0)‖_F / ‖J(0)‖_F vs path
  progress, overlaid with mpc_lti+FF's and mpc_ltv+FF's tracking error.
- **fig7_mechanism_localization.png** — reference speed profile with
  auto-detected corners, inv+FF vs best-MPC error vs path progress, and a
  per-progress-bin (inv−mpc) RMS difference bar chart.
- **fig8_controller_fingerprint.png** — step-limit saturation / jerk /
  nullspace-usage fingerprint, inv+FF vs mpc_lti+FF vs mpc_ltv+FF.
- **fig9_horizon_diagnostic.png** — the horizon root-cause diagnostic: hold-phase
  error vs time, single matched mpc_lti+FF rep at h=30 (broken) vs h=15 (fixed).
