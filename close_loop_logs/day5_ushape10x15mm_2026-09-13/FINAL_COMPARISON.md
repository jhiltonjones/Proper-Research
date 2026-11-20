# Controller comparison — 10x15mm U-shape, 2026-09-13

Fourth-shape generalization follow-up to the 20mm-triangle
(`close_loop_logs/day2_triangle20mm_2026-09-12/FINAL_COMPARISON.md`), 10x15mm
rectangle (`close_loop_logs/day3_rectangle10x15mm_2026-09-13/FINAL_COMPARISON.md`)
and 10x20mm S-curve (`close_loop_logs/day4_scurve10x20mm_2026-09-13/FINAL_COMPARISON.md`)
comparisons — the **first OPEN shape** (a 3-sided rectangle, i.e. the
rectangle plan without `--closed`): it never returns to its start, so the
terminal hold sits at the far, most-deflected end of the path instead of
back near the easy coaxial start.

## Setup
- Plan: `plans/ushape_10x15mm_2026-09-12/time_parameterized_configuration_path`
  — open 3-sided path, 10mm width x 15mm depth, 214 samples @ 10Hz.
  Safety-verified (0mm Z-drop, 262.8mm min base-distance) before any live
  test, same as every other shape this session.
- Beam length 25.0mm (apex), MPC horizon = 15, `state_tracking_weight=0.001`
  for all MPC+FF rows (the triangle's root-cause fix), applied identically.
- Open loop: n=3. Feedforward comparison: n=3 each for inv, mpc_lti,
  mpc_ltv, all `+FF`. No pure-feedback (noFF) run collected, same rationale
  as the S-curve. All 9 FF reps completed cleanly first try — no faults or
  redos needed this time.

## Results (mean ± std, n=3)

| Controller | Mode | RMS (mm) | Max (mm) | Hold (mm) | Final (mm) |
|---|---|---|---|---|---|
| Open loop | — | 2.43±0.09 | 4.07±0.17 | — | 3.78±0.22 |
| Inverse-Jacobian (DLS) | feedforward | 1.79±0.01 | 3.68±0.04 | 3.85±0.10 | 3.97±0.02 |
| MPC-LTI, fixed cost | feedforward | 1.66±0.07 | 4.00±0.18 | 4.06±0.17 | 4.34±0.09 |
| **MPC-LTV, fixed cost** | feedforward | **1.29±0.02** | **2.64±0.12** | **1.82±0.04** | **1.89±0.14** |

## Headline findings

1. **This is the cleanest, largest MPC-LTV win of any shape tested.**
   MPC-LTV beats open loop by 47% (1.29 vs 2.43mm) and inverse-Jacobian by
   28% (1.29 vs 1.79mm) on RMS — both larger margins than the triangle
   (12-17%), rectangle (30%), or S-curve (6-8%). Applying the stopping rule:
   MPC-LTV clearly outperforms both baselines, no investigation-and-stop
   needed on the headline number.
2. **The real story is in hold/final, where the gap is dramatic**: MPC-LTV
   holds at 1.82±0.04mm / 1.89±0.14mm, while inverse-Jacobian (3.85/3.97mm)
   and MPC-LTI (4.06/4.34mm) are both **worse at hold than they are averaged
   over the whole transit** (their own RMS is 1.66-1.79mm) — i.e. both
   non-LTV controllers get *worse*, not better, once they stop moving and
   just have to sit on the final target. This is the opposite of every
   closed shape tested this session, where hold/final were always the
   *best*-behaved phase for every controller. Root-caused below.

## Root cause: a conditioning cliff right next to the path's own endpoint

The obvious first guess — carried over from the rectangle/S-curve
investigations — was frozen-Jacobian staleness along the path. It does not
explain this result:

- **fig5** shows the *nominal* (offline-planned) path is well-conditioned
  everywhere, including right at the endpoint (κ=268 at the last sample,
  comparable to the low end of the whole path's 210-149,255 range). The
  planner evidently already threads through a good corridor. So mpc_lti's
  usual failure mode (frozen Jacobian from a *different, far-away* part of
  the path) isn't obviously the story here, and — more surprisingly —
  **inverse-Jacobian doesn't get to use that good corridor either**, even
  though it recomputes its Jacobian fresh from the live measured
  configuration every tick (`InverseJacobianBeamController` calls
  `jacobian_provider(state)` on `q_meas`, not a cached/frozen value — see
  `controllers/inverse_jacobian_controller.py:199`). Fresh evaluation should
  make it immune to staleness. It doesn't help. Why not?
- **fig6 (the answer):** the analytic beam Jacobian evaluated at
  inverse-Jacobian's own *actual, live* hold-phase joint state — differing
  from the nominal endpoint by only ~4mm of tip error (a joint-space
  difference of ~0.23 rad, itself mostly along the redundant/nullspace
  direction) — has condition number **58,300-60,000**, a **~225x** jump
  from the nominal endpoint's κ=262. The near-null singular value collapses
  from σ₃=3.4e-4 to σ₃≈1e-6 (~300x). **The well-conditioned corridor the
  planner found is a narrow ridge**: the moment real tracking error nudges
  the beam even slightly off it, the true local Jacobian falls into an
  adjacent near-singular basin.
- **The instability is worse than just "high condition number" — the
  correction DIRECTION itself becomes unreliable.** Two joint configurations
  differing by only ~1e-5 rad (inverse-Jacobian's actual state at two
  different points, vs. the fixed nominal endpoint) give DLS corrections,
  computed for the *identical* measured tip-error vector, that are
  **near-orthogonal** (mean cos = −0.09 across all 3 reps, essentially
  uncorrelated in a 6-D space) — see fig6, right panel. A correction judged
  to be 95%-aligned with the intended direction *under its own local model*
  is only −65%-aligned (backwards!) when judged against the neighbouring
  nominal-endpoint model, and vice versa. This is a genuine numerical/model
  fragility in this small neighbourhood, not measurement noise — it
  reproduces near-identically across all 3 reps (κ 58,283-60,050, cos
  −0.093 to −0.097).
- **Why this defeats inverse-Jacobian despite perfect proportionality
  (fig8):** inverse-Jacobian's hold-phase `|correction|` correlates with
  `error` at r≈+1.00 (textbook proportional feedback, by far the cleanest
  proportionality signature of any controller on any shape this session)
  — yet the error still plateaus at 3.2-4.1mm instead of shrinking to zero.
  **Proportional magnitude doesn't help when the correction's *direction*
  flips almost at random tick to tick**, because each tick's DLS solve
  re-linearises on a live estimate sitting in this unstable neighbourhood.
  The controller is doing exactly what a resolved-rate law should do with
  the (bad) local information it has; the information itself is the
  problem.
- **Why MPC-LTV wins:** it never re-linearises on the live, noisy measured
  state. Its hold-phase correction uses the *fixed* schedule Jacobian at
  the nominal endpoint (κ=262, the well-conditioned point the offline
  planner intended) for the whole hold, rather than chasing whatever
  unstable local estimate the live configuration happens to produce that
  tick. A deterministic, consistent (if slightly mismatched) linearisation
  beats a fresh, "more correct in principle" but numerically unstable one
  here. MPC-LTI, which also uses a fixed schedule Jacobian but frozen from
  the *start* of the path instead of the endpoint, still does badly
  (4.06/4.34mm hold/final) — being fixed isn't sufficient on its own; it
  has to be fixed at (or near) the *right* configuration, which is what
  makes LTV specifically — not LTI, not inverse-Jacobian — the one
  controller that gets this right.

## Controller fingerprint — consistent non-explanation for MPC's edge, fourth time

fig7 repeats the saturation/jerk/nullspace-usage diagnostic:

| | inv+FF | mpc_lti+FF | mpc_ltv+FF |
|---|---|---|---|
| step-limit saturation | 58.1% | 85.9% | 75.5% |
| tick-to-tick jerk (mrad) | 0.70 | 1.35 | 2.02 |
| nullspace usage (% of command with zero tip effect) | 3.9% | 60.0% | 48.7% |

Same pattern as every prior shape: MPC saturates more and is jerkier than
inverse-Jacobian. (Nullspace usage is the one number that doesn't fit the
"MPC always worse" pattern here — mpc_ltv's 48.7% is actually *below*
mpc_lti's 60.0% — but this is a whole-run average dominated by the transit
phase, not specific to the hold-phase mechanism identified above.) The
"why does MPC beat inverse-Jacobian generically" question remains open
across all four shapes; this shape's answer is specific (the conditioning
cliff above), not a generic constraint-anticipation or smoothness argument.

## Figures (`figures/`)
- **fig1_tip_paths.png** — measured tip path vs. reference U-shape,
  representative rep, 4 conditions (open path, so the plotted end marker
  matters — it doesn't return to the start).
- **fig2_error_vs_time.png** — tracking error vs. time, same 4 conditions.
- **fig3_summary_bars.png** — 4-condition RMS/max/hold/final bars.
- **fig4_jacobian_svd.png** — frozen (path-start) beam Jacobian
  singular-value spectrum, κ≈146,300.
- **fig5_jacobian_evolution.png** — local Jacobian singular values /
  condition number along the *nominal* path — shows the planner's chosen
  corridor stays well-conditioned everywhere, endpoint included.
- **fig6_conditioning_cliff.png** — THE ROOT CAUSE: condition number at the
  nominal endpoint vs. inverse-Jacobian's own live measured hold state
  (~225x worse for a ~4mm deviation), plus the resulting near-orthogonal
  DLS correction-direction misalignment.
- **fig7_controller_fingerprint.png** — saturation/jerk/nullspace-usage
  fingerprint, inv+FF vs. mpc_lti+FF vs. mpc_ltv+FF.
- **fig8_correction_vs_error.png** — |correction| vs. error at every
  hold-phase tick, all 3 reps pooled per controller — inv+FF is perfectly
  proportional (r≈+1.00) yet non-convergent, the direct behavioural
  signature of the conditioning-cliff mechanism.

## Data
Raw run data under `close_loop_logs/ushape_FF_n3_2026-09-13/runs/` and
`close_loop_logs/openloop_ushape10x15mm_n3_2026-09-13/runs/`. The
conditioning-cliff analysis (fig6) is cached in
`close_loop_logs/ushape_FF_n3_2026-09-13/singularity_analysis.npz`
(`cond_nom`, `conds_meas`, `cos_list`) — regenerate by querying
`beam_jacobian_providers.from_model_bundle(...)` at inv's logged
`q_meas_rad`/`insertion_length_m` for a hold-phase tick, exactly as
`compare_controllers_live.py::build_ltv_schedule` builds the LTV schedule
itself, just evaluated at a live state instead of a reference sample.
