# Controller comparison — 10x15mm rectangle, 2026-09-12/13

Shape-generalization follow-up to the 20mm-triangle comparison
(`close_loop_logs/day2_triangle20mm_2026-09-12/FINAL_COMPARISON.md`). Same
methodology, same fixed-MPC configuration, a different (closed, 4-corner)
shape.

## Setup
- Plan: `plans/rectangle_10x15mm_skipglobal_2026-09-12/time_parameterized_configuration_path` — 10mm width (lateral) x 15mm depth (axial), closed, 265 samples @ 10Hz, 26.4s. Built with the new Z-no-decrease + magnet-to-beam-base exclusion safety constraints (see the `beam-lateral-authority-limit` memory, 2026-09-12 entries) and via `--skip-global` (Layer 1 alone was already 100% node-feasible; skipping Layer 2 avoided the sharp-corner global-optimizer convergence issues a larger 20x30mm rectangle attempt hit).
- Beam length 25.0mm (apex), MPC horizon = 15, `state_tracking_weight=0.001` for all MPC+FF rows (the root-cause fix from the triangle investigation) — applied identically here, not re-derived.
- Open loop: n=1 on each of two separate days (2026-09-12, 2026-09-13) rather than n=3 same-day — reported as a range, not averaged with a std.
- Pure-feedback and feedforward rows: n=3 each, one subprocess-crash retry apiece (`rc=4`, known RTDE reconnect fragility), both recovered cleanly on retry.

## Results (mean ± std, n=3 except open loop)

| Controller | Mode | RMS (mm) | Max (mm) | Hold (mm) | Final (mm) |
|---|---|---|---|---|---|
| Open loop | — | 2.51–2.88 | 5.24–5.86 | 1.77–1.96 | 1.68–1.88 |
| Inverse-Jacobian (DLS) | pure feedback | 6.34±0.05 | 10.84±0.11 | 0.15±0.08 | 0.17±0.13 |
| MPC-LTI | pure feedback | 6.78±0.01 | 11.28±0.04 | 0.17±0.04 | 0.18±0.03 |
| MPC-LTV | pure feedback | 6.96±0.04 | 11.37±0.18 | 0.10±0.04 | 0.10±0.03 |
| Inverse-Jacobian (DLS) | feedforward | 2.54±0.21 | 6.37±0.92 | 0.77±0.18 | 0.72±0.32 |
| MPC-LTI, fixed cost | feedforward | 2.66±0.05 | 7.20±0.14 | 0.55±0.04 | 0.33±0.19 |
| **MPC-LTV, fixed cost** | feedforward | **1.77±0.03** | **4.50±0.08** | 0.50±0.14 | 0.26±0.08 |

## Headline findings

1. **Every controller loses to open loop in pure feedback** (6.3–7.0mm vs
   ~2.5–2.9mm) — identical to the triangle. The offline plan is simply a
   better answer than any online re-derivation, regardless of shape.
2. **In pure feedback the ranking is the *opposite* of feedforward**:
   inverse-Jacobian is best on RMS/max (6.34mm), MPC-LTV is worst
   (6.96mm) — but MPC-LTV is *tightest* on hold/final (0.10mm vs
   inverse-Jacobian's 0.15–0.17mm). Stds are small (0.01–0.18mm across the
   board), so this inversion is a real, repeatable effect, not noise. MPC's
   advantage lives entirely in the feedforward architecture, not in its
   control law being generically better.
3. **In feedforward, MPC-LTV is decisively the best controller — a 30% RMS
   improvement over inverse-Jacobian (1.77 vs 2.54mm)**, closely matching
   the triangle's margin, confirming the fixed-MPC result generalizes to a
   second, independently-designed shape rather than being triangle-specific.
4. **MPC-LTI does *not* beat inverse-Jacobian here** (2.66 vs 2.54mm,
   slightly worse) — different from the triangle, where LTI edged narrowly
   ahead too. Only the per-step-relinearized LTV variant shows a real gain
   on this shape.

## Why LTV beats LTI here — same mechanism as the triangle

fig5/fig6 reproduce the triangle's Jacobian-evolution analysis for this
shape's own per-step relinearized schedule:

- **fig5** — the rectangle's local Jacobian conditioning swings from the
  frozen (sample-0) value (κ≈75,000) up to a broad peak (κ≈150,000) around
  the first corner, down to a much better-conditioned plateau (κ≈200–300)
  along the long straight edge in the middle of the path, then back up
  again approaching the closing corner. Milder than the triangle's
  near-exact singularities (κ up to 700,000+), but the same qualitative
  story: the frozen linearization is wrong in both directions, not just
  "somewhat stale."
- **fig6** — directly ties this to tracking error: where the frozen-Jacobian
  mismatch peaks a second time (~170% of ‖J(0)‖, around sample 200–240,
  approaching the far corner), MPC-LTI's error spikes to **7.3mm** while
  MPC-LTV's stays at **4.5mm** on the same stretch. Clean, visible,
  reproduces the triangle's finding on an independent shape.

## Why MPC (LTV) beats inverse-Jacobian — same honest non-explanation as the triangle

fig7 repeats the "controller fingerprint" check from the triangle (does MPC
win because it anticipates constraints better, or because its commands are
smoother?) — and gets the **same negative result**:

| | inv+FF | mpc_lti+FF | mpc_ltv+FF |
|---|---|---|---|
| step-limit saturation | 48.8% | 71.9% | 62.8% |
| tick-to-tick jerk (mrad) | 0.69 | 1.72 | 1.89 |
| nullspace usage (% of command with zero tip effect) | 5.7% | 57.4% | 54.3% |

MPC saturates *more*, is *jerkier*, and wastes *more* of its command in the
exact nullspace of the Jacobian than inverse-Jacobian — on this shape too.
Whatever gives MPC-LTV its edge, it is not better constraint anticipation or
smoother commands, exactly as found on the triangle. Still an open question.

## A new, important finding this shape surfaced: the state_tracking_weight fix does not fully generalize

This is the one result from this comparison that is **not** a clean repeat
of the triangle, and needs to be reported plainly rather than smoothed over.

fig8 checks whether each controller's correction magnitude tracks the
actual residual error during the terminal hold — the same diagnostic that
originally found and confirmed the state_tracking_weight bug on the
triangle (pre-fix: r=−0.85 to −0.95; post-fix: r=+0.77 to +0.90).

**On the rectangle:**
- **inverse-Jacobian+FF is cleanly proportional in every rep**: r=+1.00,
  +1.00, +0.96 — textbook feedback behaviour, correction scales with error
  from near-zero up to ~2mm.
- **MPC-LTV+FF is *negatively* correlated in every rep**: r=−0.45, −0.28,
  −0.96 — the correction sits pinned in a narrow 0.18–0.22 rad band
  regardless of whether the actual error is 0.05mm or 1.3mm. This is the
  same qualitative signature as the original pre-fix bug on the triangle,
  now appearing on the rectangle **despite the fix being applied**
  (`state_tracking_weight=0.001`, identical config used for both shapes).

**Why this didn't show up as a bad RMS/hold number:** the pinned correction
magnitude here (~0.20 rad) happens to be roughly the right order for this
shape's residuals, so the practical damage is smaller than the original
triangle bug's — hold/final (0.50/0.26mm) still look fine in the aggregate
table. But the *mechanism* generating that number is not behaving as a
sound proportional controller; it is coincidentally landing in a reasonable
place, not correctly computing it from the current error the way
inverse-Jacobian's DLS law demonstrably does.

**Root-caused: this is a second, independent, horizon-dependent effect —
not the same mechanism as the triangle's `state_tracking_weight` bug.**
Repeating the triangle's exact diagnostic (mpc_ltv+FF at horizon=1, DARE
off, `state_tracking_weight=0.001` kept) on the rectangle
(`close_loop_logs/rectangle_ltv_h1_2026-09-13/`):

| | horizon=1 | horizon=15 (the n=3 comparison above) |
|---|---|---|
| RMS | 1.77mm | 1.77±0.03mm (identical) |
| corr(\|correction\|, error), hold | **+0.87** | −0.28 to −0.96 |
| cos(mpc, counterfactual-DLS) | **+0.49 mean, 11% negative** | not directly measured, but consistent with the negative hold correlation above |
| nullspace-routed fraction | 0.49 (hold), 0.53 (whole-run) | 0.54 (whole-run, from fig7) — essentially unchanged |

RMS is identical at both horizons — but the hold-phase correlation
**completely flips sign** between horizon=1 (strongly positive, proportional)
and horizon=15 (strongly negative, pinned/non-proportional). Nullspace
routing stays about the same either way, so that's not the differentiator.
**The 15-step lookahead itself — not the state-tracking term, which is
already at its fixed value in both tests — is reintroducing non-proportional
hold behaviour on this shape.** This is a genuinely different mechanism from
the triangle's bug (a pure temporal-history/cost-structure issue there, with
no horizon dependence once fixed): here, the rectangle's sharper local
Jacobian conditioning swings (fig5) most likely interact with a 15-step
receding-horizon plan in a way a 9.4mm-lateral triangle's gentler geometry
did not expose.

**Bisected: the flip happens between h=1 and h=5, and it is specifically a
hold-phase phenomenon.** Testing h=5 (`close_loop_logs/rectangle_ltv_h5_2026-09-13/`):

| | h=1 | h=5 | h=15 (n=3 comparison) |
|---|---|---|---|
| RMS | 1.77mm | 1.77mm | 1.77±0.03mm — **identical at every horizon tested** |
| Hold-only correlation | +0.87 | **−0.85** | −0.28 to −0.96 |
| Whole-run correlation | +0.65 | +0.58 | not measured |
| cos(mpc, counterfactual-DLS) | +0.49 mean, 11% negative | +0.40 mean, 12% negative | consistent with the negative hold correlation |

At h=5 the hold-only correlation has **already flipped fully negative**
(−0.85, matching h=15's severity), even though the whole-run cosine
similarity still looks reasonable (+0.40, close to h=1's +0.49). So the
pathology is confined to the **hold phase specifically**: once the target
stops moving, even a 5-step lookahead is enough for the QP to start
optimizing a smoothed multi-step settle-to-zero plan instead of reacting to
the instantaneous residual, while the transit phase (moving target) stays
well-behaved at any horizon tested. The exact boundary (h=2, h=3, or h=4)
was not narrowed further, since it wouldn't change the practical answer
below.

**Practical implication and recommendation:** since RMS is bit-for-bit
identical across every horizon tested (h=1, 5, and 15 all give 1.77mm),
there is **no accuracy cost** to using h=1 on this shape, and it is the only
tested horizon with a genuinely sound (positive, proportional) hold-phase
control law. **h=1 is the recommended configuration for mpc_ltv+FF on this
rectangle**, not h=15 — the longer horizon buys nothing here and actively
introduces an unsound (if numerically small-in-effect) hold-phase behaviour.
Whether the triangle has a similar, currently-undetected horizon
sensitivity at some length beyond 15 remains an open question.

## Figures (`figures/`)
- **fig1_tip_paths.png** — measured tip path vs. reference rectangle, representative rep, 4 conditions.
- **fig2_error_vs_time.png** — tracking error vs. time, same 4 conditions, aligned to t=0.
- **fig3_summary_bars.png** — full 7-condition RMS/max/hold/final bars.
- **fig4_jacobian_svd.png** — frozen beam Jacobian singular-value spectrum (κ≈145,600).
- **fig5_jacobian_evolution.png** — local Jacobian conditioning along the path (LTV schedule vs. LTI's frozen value).
- **fig6_jacobian_mismatch_vs_error.png** — frozen-Jacobian mismatch vs. progress, overlaid with LTI's/LTV's tracking error.
- **fig7_controller_fingerprint.png** — saturation/jerk/nullspace-usage fingerprint, inv+FF vs. mpc_lti+FF vs. mpc_ltv+FF.
- **fig8_correction_vs_error.png** — |correction| vs. error at every hold-phase tick, all 3 reps pooled per controller — the negative-correlation finding above.

## Data
Raw run data under `close_loop_logs/rectangle10x15mm_{FF,noFF}_n3_2026-09-13/runs/` and `close_loop_logs/openloop_rectangle10x15mm_2026-09-{12,13}/runs/`.
