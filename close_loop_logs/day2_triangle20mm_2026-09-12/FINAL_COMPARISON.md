# Controller comparison — 20mm triangle, 2026-09-12

## Setup
- Plan: `plans/triangle_20mm_2026-09-11/time_parameterized_configuration_path` (20mm base / 14mm depth apex-at-start triangle, apex insertion 25mm, max insertion ~40mm, 223 samples @ 10Hz, 22.2s duration, built at 0.95x velocity/acceleration safety factor — deliberately time-optimal).
- Beam length 25.0mm (apex), all runs on the same day/calibration state.
- MPC horizon = 15 for both `mpc_lti` (frozen Jacobian at sample 0) and `mpc_ltv_offline` (per-sample relinearized schedule), chosen as a feasible/solve-time-safe horizon established the previous day.
- RMS/max computed over the transit phase only (`terminal_hold == False`); hold = mean error during the terminal hold; final = error at the last control step.
- n=3 per condition except where noted; failed subprocess attempts (RTDE reconnect fragility — `rc=4`/`rc=-11` segfaults, retried automatically, 2 occurrences total) excluded from aggregation (retry succeeded in every case).

## Root-cause fix applied to the feedforward MPC rows
`ConfigurationMPCConfig.state_tracking_weight` (default 1.0, `state_error_scale`=0.5°/joint ⇒ effective weight ≈13,000/rad²) adds a joint-space "catch up to the planned trajectory" cost that is redundant with, and uncoordinated against, feedforward's own direct joint-state servoing (`q_target = ref_state[target_index] + trim`). It was empirically confirmed (via a counterfactual comparison against plain DLS on identical measured errors) to be the reason every earlier MPC+FF attempt this session — regardless of horizon, `Rd`, or DARE tuning — was worse than open loop. Fix: `state_tracking_weight = 0.001` (three orders of magnitude down — exact zero breaks the DARE terminal-cost solve, since the terminal weight `Q + Jᵀ·Qp·J` needs `Q`'s full rank to stay detectable given the beam's rank-≤3-in-7D Jacobian). Full derivation in the `beam-lateral-authority-limit` memory file, entries dated 2026-09-12.

## Results (mean ± std, n=3)

| Controller | Mode | RMS (mm) | Max (mm) | Hold (mm) | Final (mm) |
|---|---|---|---|---|---|
| Open loop | — | 2.26 ± 0.11 | 4.00 ± 0.52 | 1.04 ± 0.67 | 0.99 ± 0.68 |
| Inverse-Jacobian (DLS) | pure feedback | 4.53 ± 0.01 | 9.73 ± 0.23 | 0.27 ± 0.16 | 0.30 ± 0.23 |
| Inverse-Jacobian (DLS) | feedforward | **1.76 ± 0.03** | 3.51 ± 0.24 | 0.53 ± 0.11 | 0.60 ± 0.21 |
| MPC-LTI, default cost | pure feedback | 4.60 ± 0.03 | 10.01 ± 0.25 | 0.41 ± 0.07 | 0.41 ± 0.11 |
| MPC-LTI, fixed cost (`state_tracking_weight=0.001`) | feedforward | **1.65 ± 0.05** | **3.46 ± 0.16** | 0.34 ± 0.06 | 0.29 ± 0.04 |
| MPC-LTV, default cost | pure feedback | 4.71 ± 0.02 | 10.15 ± 0.27 | 0.24 ± 0.13 | 0.14 ± 0.06 |
| MPC-LTV, fixed cost (`state_tracking_weight=0.001`) | feedforward | **1.59 ± 0.04** | **2.98 ± 0.08** | 0.35 ± 0.08 | 0.45 ± 0.16 |

## Headline findings
1. **Pure feedback loses to open loop on every controller** (4.5-4.7mm vs 2.26mm RMS) — re-deriving the joint trajectory from a locally-linearized model every 100ms cannot match the ~770s offline nonlinear multistart solve it's competing against.
2. **Feedforward beats open loop on every controller** (1.6-1.8mm vs 2.26mm, a 22-30% RMS reduction) by reusing the expensive offline solution and only correcting the residual.
3. **MPC-LTV+FF (fixed) is the best controller overall** — 1.59±0.04mm RMS, 2.98±0.08mm max — narrowly ahead of MPC-LTI+FF (1.65±0.05mm) and inverse-Jacobian+FF (1.76±0.03mm), with the tightest max-error spread of any condition.
4. This ranking was **only achievable after fixing the `state_tracking_weight` leak** — before the fix, MPC+FF was the worst controller tested all session (2.8-3.0mm RMS, worse than open loop), diagnosed via a full root-cause chain: horizon=1 isolation → counterfactual DLS comparison → code-level trace to the base MPC cost class → targeted fix → DARE-solvability regression → small-weight resolution → n=3 confirmation on both MPC variants.

## Diagnostic signature of the fix (see memory for full per-rep numbers)
- Before fix: corr(|correction|, tracking error) during the terminal hold was **negative** (−0.78 to −0.95) for both MPC variants — the correction shrank as error grew and vice versa, an inverted, non-proportional control signature.
- After fix: hold-phase correlation is **strongly positive** for both variants (mpc_lti: +0.82 to +0.87; mpc_ltv: +0.84 to +0.90), matching the physically-sound proportional behaviour inverse-Jacobian's DLS law has always shown (+0.996).
- `directional_damping` (anisotropic near-null-direction penalty) was tested on top of the fixed mpc_lti+FF baseline: it reliably cuts weak-direction routing at hold from 83% to 20-30% and keeps the correlation positive, but does not improve (and modestly worsens) net RMS/max on this shape — validated-but-not-yet-beneficial, not deployed.

## Follow-up: what actually explains MPC's small RMS edge (fig9's open question)
fig9 showed the two intuitive explanations for MPC's edge — better constraint
anticipation, smoother commands — are both contradicted by the data (MPC
saturates more, is jerkier, wastes more command in the Jacobian's null space).
A deeper per-tick investigation (see `figures/fig10_terminal_hold_anticipation.png`)
found:

1. **It isn't a better model.** `inverse_jacobian_controller.py` (`naive_inverse_jacobian`,
   lines ~198-200) re-evaluates the Jacobian from the live measured joint state
   every tick. `mpc_variants.precompute_schedule` (lines ~300-324) evaluates the
   Jacobian once, offline, indexed by the *planned* reference state — never the
   measured one (`mpc_lti` freezes sample 0 of that schedule; `mpc_ltv_offline`
   uses the full offline schedule but still indexed by planned progress). MPC's
   model is structurally staler than inverse-Jacobian's, so a fresher/better
   local linearization cannot be the source of its edge.
2. **It isn't the two geometric corners either.** Using the reference plan's own
   tangent vectors (`plans/triangle_20mm_2026-09-11/.../time_parameterized_configuration_path.csv`),
   the two sharp corners are at `sample_index` ≈80 and ≈144. Binning tracking
   error by path progress (representative median-RMS reps: `inv_ff_reps23/runs/inv_rep1_20260912T101745Z`
   vs `mpc_ltv_ff_h15_dare_confirm_n3/runs/ltv_rep1_20260912T100948Z`), inverse-Jacobian+FF
   is *marginally better* right around both corners (progress bins spanning
   66-133: diff (inv−ltv) is negative, i.e. inv wins, by 0.01-0.31mm).
3. **The gap concentrates after the 2nd corner**, in the decel-out-of-turn /
   reaccelerate-on-return-leg / abrupt-full-stop transient (`ref_index` ≈155-222,
   see fig10's middle panel for the reference speed profile — this is *not* a
   steady-speed cruise, so "MPC anticipates a steady segment" isn't the story
   either). In that stretch, ltv's error runs 0.32-0.83mm below inv+FF's per
   progress-bin — this is where essentially all of the aggregate RMS gap comes
   from; bins covering `ref_index` 133-222 average +0.30mm (inv−ltv) per tick,
   vs a roughly net-zero/slightly-negative diff over `ref_index` 0-133.
4. **Statistical read:** the n=3 aggregate gap (inv+FF 1.76±0.03mm vs mpc_ltv+FF
   1.59±0.04mm RMS) is ~3-4 pooled standard deviations apart on this shape —
   probably a real effect, though this was only confirmed in detail on one
   representative rep's per-tick trace, not all 6 reps pooled at the tick level.
5. **Still open:** whether this is really MPC's 15-step horizon smoothing out a
   changing-speed reference segment that inverse-Jacobian's single-step reactive
   law can't (the mechanistically plausible remaining explanation) was not
   confirmed at the command level — that requires projecting each tick's applied
   correction onto the Jacobian's row space vs. null space, and the raw per-tick
   Jacobian matrices aren't logged (only a scalar `jacobian_condition`). Treat
   the *location* of the edge (post-2nd-corner decel/reaccel transient) as
   established; the *mechanism* at the command level as a documented but
   unconfirmed hypothesis.

## Data
Every run's `path_follow.jsonl` (per-tick diagnostics), `tip_trajectory.csv`, `frozen_jacobian.json`, `path_follow_plot.png` and `summary.json` are under `close_loop_logs/day2_triangle20mm_2026-09-12/*/runs/` and `close_loop_logs/openloop_triangle20mm_2026-09-12/runs/` (open-loop rep 1) + `.../openloop_reps23/runs/` (reps 2-3).

## Figures (`figures/`)
- **fig1_tip_paths.png** — measured tip path vs. reference triangle, one representative (median-RMS) rep for open loop / inv+FF / mpc_lti+FF (fixed) / mpc_ltv+FF (fixed), in the beam plane (B.y horizontal, B.x vertical).
- **fig2_error_vs_time.png** — tracking error vs. time (aligned to each run's own t=0) for the same four conditions overlaid — shows open loop consistently above the three feedforward controllers through both corner-approach peaks, converging together at the hold.
- **fig3_summary_bars.png** — full 7-condition grouped bar chart (RMS / max / hold / final, mean±std, n=3, hatched=pure feedback, solid=feedforward).
- **fig4_jacobian_svd.png** — singular-value spectrum of the frozen beam Jacobian (log scale), showing the near-null third direction (σ₃≈5.9×10⁻⁷ vs σ₁≈0.086, σ₂≈0.037; condition number ≈146,000) responsible for the beam's weak lateral steering authority.
- **fig5_weak_direction_routing.png** — % of hold-phase joint correction routed through the Jacobian's near-null direction: inverse-Jacobian+FF (~0%) vs. mpc_lti+FF pre-fix (64%) vs. mpc_lti+FF fixed, no directional damping (84% — *worse* than pre-fix, since `state_tracking_weight` had incidentally been providing some broad-spectrum damping too) vs. mpc_lti+FF fixed + `directional_damping=0.1` (26%). Makes the case that the temporal fix (`state_tracking_weight`) and the spatial fix (`directional_damping`) address genuinely independent axes.
- **fig6_correction_vs_error_before_after.png** — the root-cause diagnostic: |joint correction| vs. tracking error at every hold-phase tick, mpc_lti+FF pre-fix (correction pinned ~0.213 rad regardless of error, corr=−0.85) vs. fixed (correction scales cleanly with error, corr=+0.86).
- **fig7_jacobian_evolution.png** — singular values and condition number of the LOCAL beam Jacobian at every reference sample (mpc_ltv_offline's relinearised schedule) vs. path progress, with mpc_lti's frozen (sample-0) value overlaid as a dotted reference. The near-null direction (σ₃) crashes to a near-exact singularity (condition number >700,000) at the triangle's two sharp corners, while away from the corners the beam is actually *far better* conditioned than the frozen value assumes (κ~300-500 vs the frozen ~76,000) — the frozen Jacobian is wrong in both directions, not just "somewhat stale."
- **fig8_jacobian_mismatch_vs_error.png** — Frobenius-norm deviation of the local Jacobian from the frozen (sample-0) one, growing to 240% of ‖J(0)‖ at its peak, plotted above mpc_lti+FF's and mpc_ltv+FF's tracking error along the same path-progress axis — the regions of largest mismatch are visibly where LTI's error runs above LTV's.
- **fig9_controller_fingerprint.png** — an honest negative-result diagnostic: step-limit saturation frequency, tick-to-tick command jerk, and exact-nullspace usage (fraction of command magnitude with literally zero effect on the tip) for inv+FF vs. the two fixed MPC variants. Both intuitive explanations for MPC's small RMS edge over inverse-Jacobian ("it anticipates constraints better," "its commands are smoother") are **contradicted** by this data — MPC saturates more (62-78% vs 46%), is jerkier (2.0-2.5 vs 1.0 mrad/tick), and wastes over half its command magnitude in directions with zero tip effect (54-57% vs 2%). The mechanism behind MPC's edge is not pinned down by these checks and is noted as an open question rather than overclaimed.

- **fig10_terminal_hold_anticipation.png** — follow-up diagnostic: tracking
  error vs. path progress for inv+FF and mpc_ltv+FF (fixed) with the two real
  geometric corners marked (from the plan's own tangent vectors, not a noisy
  vision-derived estimate), the reference path speed profile (showing the
  decel/reaccel transient after the 2nd corner, not a steady cruise), and a
  per-bin (inv−ltv) error-difference bar chart. Shows the RMS gap is *not* at
  the corners and concentrates entirely in the post-2nd-corner transient.
