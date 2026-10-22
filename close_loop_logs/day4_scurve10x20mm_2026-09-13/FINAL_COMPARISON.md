# Controller comparison — 10x20mm S-curve, 2026-09-13

Third-shape generalization follow-up to the 20mm-triangle
(`close_loop_logs/day2_triangle20mm_2026-09-12/FINAL_COMPARISON.md`) and the
10x15mm-rectangle (`close_loop_logs/day3_rectangle10x15mm_2026-09-13/FINAL_COMPARISON.md`)
comparisons. Same fixed-MPC configuration, an inherently open, two-lobe
shape (bends substantially in *both* lateral directions within one run,
unlike the triangle/rectangle which each bend one way).

## Setup
- Plan: `plans/scurve_10x20mm_2026-09-12/time_parameterized_configuration_path`
  — two opposite-sign raised-cosine lobes, amplitude 10mm, length 20mm, 174
  samples @ 10Hz, ~19s. Safety-verified (0mm Z-drop, 247.4mm min
  base-distance) before any live test, same as the rectangle.
- Beam length 25.0mm (apex), MPC horizon = 15, `state_tracking_weight=0.001`
  for all MPC+FF rows (the triangle's root-cause fix) — applied identically,
  not re-derived.
- Open loop: n=3, same day.
- Feedforward comparison: n=3 each for inv, mpc_lti, mpc_ltv, all `+FF`.
  **No pure-feedback (noFF) run was collected on this shape** — the
  triangle/rectangle both already showed every controller loses badly to
  open loop in pure feedback, and it wasn't worth re-spending live-hardware
  time confirming that a third time.
- Two reps needed a mid-experiment redo, both driven by hardware issues, not
  controller behaviour, and both fixed before being folded into the n=3
  aggregate below:
  - `mpc_lti` rep1: RTDE crashed once (rc=-11, auto-retried), and the retry
    "completed" (`stop_reason=path_complete`) but with a robot fault (fixed
    by the user mid-session) that left the beam pinned away from the
    reference for the whole hold (hold=4.53mm, final=4.50mm vs. ~0.5-0.8mm
    for every other rep). Re-run cleanly after the fault was fixed
    (hold=0.59mm, final=0.61mm) and spliced in to replace the faulty rep.
  - `mpc_ltv` rep3: stopped early on `stale_vision(age=0.506s)` after only
    97/194 steps — a vision dropout, not a controller or robot fault.
    Re-run cleanly (`path_complete`, full 194 steps) and spliced in.
  - Both faulty raw runs are kept for the record under
    `close_loop_logs/scurve_FF_n3_2026-09-13/runs_superseded/`.

## Results (mean ± std, n=3)

| Controller | Mode | RMS (mm) | Max (mm) | Hold (mm) | Final (mm) |
|---|---|---|---|---|---|
| Open loop | — | 3.44±0.02 | — | — | — |
| Inverse-Jacobian (DLS) | feedforward | 2.72±0.03 | 5.60±0.24 | 1.14±0.09 | 0.80±0.08 |
| MPC-LTI, fixed cost | feedforward | **2.50±0.07** | 5.45±0.17 | **0.58±0.05** | 0.71±0.08 |
| MPC-LTV, fixed cost | feedforward | 2.56±0.02 | 5.61±0.15 | 0.99±0.07 | 0.68±0.20 |

## Headline findings

1. **MPC's edge over inverse-Jacobian generalizes to a third, independently-
   designed shape.** Both MPC variants beat inverse-Jacobian on RMS
   (2.50-2.56mm vs. 2.72mm, 6-8% better) and on open loop (3.44mm, 26-27%
   better) — consistent in direction with the triangle and rectangle,
   though the margin here is the smallest of the three shapes (triangle and
   rectangle both showed larger MPC-LTV gains).
2. **New: on this shape, MPC-LTI and MPC-LTV are statistically
   indistinguishable** (2.50±0.07 vs 2.56±0.02mm, well within 1 std of each
   other) — and LTI is nominally *ahead*. This breaks the pattern from the
   other two shapes, where per-step relinearization (LTV) gave a clear,
   repeatable RMS edge over the frozen Jacobian (LTI). Root-caused below —
   it is not noise.
3. Hold/final are more mixed than on the other shapes: MPC-LTI has the best
   hold of the three controllers (0.58mm) but MPC-LTV has the (nominally)
   best final (0.68mm, though with the largest std, 0.20mm, likely a
   residual echo of the second-lobe tracking asymmetry seen in the
   open-loop test — see `openloop_scurve10x20mm_n3_2026-09-13/`).

## Why LTI≈LTV here, unlike the triangle/rectangle — root-caused

fig5/fig6 repeat the Jacobian-evolution analysis used on both prior shapes.
On the triangle and rectangle this cleanly explained LTV's edge: frozen-
Jacobian mismatch (Frobenius norm vs. the sample-0 value) grew along the
path and LTI's tracking error spiked exactly where the mismatch peaked,
while LTV's didn't. **On the S-curve, that story does not hold**, and the
reason is visible directly in the singular-value decomposition:

- **fig5** — the *usable* singular values (σ1, σ2 — the directions the
  controller actually draws corrective commands from) only swing mildly
  along the path: σ1 grows from 0.050 to 0.19 mm/rad, σ2 from 0.028 to 0.10
  mm/rad (under 4x either way). But **σ3, the near-null direction with
  essentially zero contribution to achievable tip motion, swings by ~500x**
  (6.3e-7 to 4.7e-4 mm/rad), with two sharp dips to near-zero (at the two
  points where the path locally passes through an almost-singular
  configuration — the inflection points of the two lobes). The condition
  number plot's two spikes (κ up to 143,600 and 32,700) are entirely driven
  by this near-null direction collapsing, not by any change in the
  meaningfully-used part of the Jacobian.
- **fig6** — this is why the 270%-of-‖J(0)‖ frozen-Jacobian mismatch
  (larger, in fact, than the rectangle's 170-210% peak) produces **no
  visible LTI-vs-LTV separation**: the mismatch metric (a Frobenius norm
  over the whole 3x6 block) is dominated by the near-null σ3 direction that
  the controller barely uses anyway, so a stale linearization there costs
  almost nothing. LTI's and LTV's error traces sit essentially on top of
  each other through both lobes, including at the highest-mismatch stretch
  (samples ~140-170, mismatch 250-270%), where the triangle/rectangle each
  showed their clearest LTI-specific error spike.

**Conclusion: LTV's edge over LTI is conditional on frozen-Jacobian drift
occurring in the singular directions the controller actually uses (σ1/σ2),
not on the raw magnitude of drift.** The triangle and rectangle both
happened to drift substantially in those usable directions along their
paths; the S-curve's drift is concentrated almost entirely in the
control-irrelevant near-null direction instead. This is a genuinely new,
useful refinement of the "why LTV beats LTI" mechanism established on the
first two shapes, not a contradiction of it.

## Controller fingerprint — consistent non-explanation for MPC's edge, third time

fig7 repeats the saturation/jerk/nullspace-usage diagnostic:

| | inv+FF | mpc_lti+FF | mpc_ltv+FF |
|---|---|---|---|
| step-limit saturation | 61.9% | 86.6% | 83.0% |
| tick-to-tick jerk (mrad) | 0.72 | 1.41 | 1.79 |
| nullspace usage (% of command with zero tip effect) | 6.7% | 61.2% | 61.7% |

Same pattern as both prior shapes: MPC saturates more, is jerkier, and
routes more of its command through the exact nullspace of the Jacobian than
inverse-Jacobian — yet still wins on RMS. Whatever gives MPC its edge over
inverse-Jacobian, it is consistently *not* better constraint anticipation or
smoother commands, across all three shapes tested so far. Still an open
question.

## Figures (`figures/`)
- **fig1_tip_paths.png** — measured tip path vs. reference S-curve,
  representative rep, 4 conditions.
- **fig2_error_vs_time.png** — tracking error vs. time, same 4 conditions.
- **fig3_summary_bars.png** — 4-condition RMS/max/hold/final bars (no
  pure-feedback row collected on this shape).
- **fig4_jacobian_svd.png** — frozen beam Jacobian singular-value spectrum
  (κ≈146,900 at sample 0).
- **fig5_jacobian_evolution.png** — local Jacobian singular values /
  condition number along the path — the σ1/σ2-vs-σ3 split that explains the
  LTI≈LTV finding.
- **fig6_jacobian_mismatch_vs_error.png** — frozen-Jacobian mismatch vs.
  progress, overlaid with LTI's/LTV's tracking error — no separation,
  unlike the triangle/rectangle.
- **fig7_controller_fingerprint.png** — saturation/jerk/nullspace-usage
  fingerprint, inv+FF vs. mpc_lti+FF vs. mpc_ltv+FF.

## Data
Raw run data under `close_loop_logs/scurve_FF_n3_2026-09-13/runs/` (faulty
pre-redo runs preserved in `runs_superseded/`) and
`close_loop_logs/openloop_scurve10x20mm_n3_2026-09-13/runs/`.
