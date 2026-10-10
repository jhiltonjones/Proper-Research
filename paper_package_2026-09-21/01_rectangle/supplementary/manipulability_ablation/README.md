# Manipulability/authority-shaping MPC objective — offline ablation (closed, 2026-09-22)

Status: **closed research branch, negative/null result.** The validated live
controller (`StagewiseTaskNullspaceDelayAwareMPC`, the `Q_N` reference-nullspace
regularizer used everywhere else in this package) was never modified. This
directory is the full record of why an alternative secondary objective was
tried, and why it was not adopted.

## One-line summary for the paper

> We also investigated replacing reference-nullspace regularization with a
> projected log-volume authority reward. Although the nonlinear authority
> metric possessed exploitable nullspace gradients along the nominal
> trajectory, its reference-linearized first-order surrogate frequently
> mispredicted the sign of the true authority change in replay and increased
> tracking error and control effort as its weight increased. We therefore
> retained the reference-nullspace regularizer.

## What was tried

Secondary objective `φ(z) = logdet(I + J̄J̄ᵀ) = Σᵢ log(1+σᵢ²)` ("regularized
log-volume authority utility" — diminishing-return, not a singularity
barrier), computed on the beam-plane-projected, physically-scaled Jacobian
`J̄ = S_p⁻¹·J_task(z)·D_u·dt`. Proposed as a drop-in replacement for the
validated `Q_N = ‖P_N S_z⁻¹(z − z_ref)‖²` nullspace-posture anchor, gated by
a strict Phase 0 → conflict-diagnostic → QP-integration → regression-test →
A/B/C-replay sequence (scripts in `scripts/`, raw results in `data/`).

Three QP modes were built into `authority_nullspace_mpc.py` (kept as research
infrastructure, not wired into any live run script):
- **A** `reference_nullspace` — bit-identical reproduction of the validated
  controller (regression Test A: max command diff < 1e-10 across all tested
  ticks).
- **B** `none` — no secondary state cost at all. Essential control condition:
  distinguishes "removing the anchor helped" from "the new objective helped."
- **C** `manipulability` — replaces (does not add to) the anchor with a
  linear authority-ascent term in `f` only, `H` unchanged from B.

## Phase 0 (diagnostic gate) — passed

`g_N` (nullspace-projected authority gradient) was appreciable on 85% of the
265-point rectangle reference path; the true nonlinear φ increased along a
small step in that direction at 93% of points (meaningful at 76%). The
underlying nonlinear authority landscape is genuinely exploitable — this is
not a "there was nothing there" result.

## Conflict diagnostic

Compared against real MPC-SJ live tracking deviation (5 rectangle runs), the
existing `Q_N` anchor direction and the authority-ascent direction are
genuinely, path-position-dependently opposed in places (concentrated in the
low-φ tail of the path), not merely uncorrelated. Confirms there was a real
question worth testing, not a strawman.

## Weight calibration pilot

A naive global `‖f_manip‖/‖f_rest‖` norm ratio (which conflates
task-relevant and nullspace-relevant cost directions) suggested λ≈10³–10⁴.
A cheap 10-snapshot × 7-λ solution-sensitivity pilot showed this was
**almost two decades too high** at ordinary path points — λ=30 already gave
82–91% of full command-scale saturation. Corrected, solution-sensitivity-based
bracket: **λ ∈ {1, 3, 10, 30}**, spanning weak (~3–4%) → medium/strong
(~12–15%) → strong (~43–52%) → aggressive/near-saturated (~82–91%) command
change relative to λ=0.

## A/B/C replay — the decisive result

Single-tick counterfactual replay (same technique as the insertion-limit
sweep elsewhere in this package: re-solve each mode's QP against the actual
recorded state/previous-input/measured-tip at each real historical tick, so
trajectories don't diverge/compound — isolates each mode's one-step decision
cleanly): 175 ticks × 6 modes (A, B, C at 4 λ) = 1050 solves, 1050 true-φ
evaluations, across the same 5 real MPC-SJ rectangle live logs used
throughout this package.

**True authority got worse, not better, and monotonically so with weight:**

| mode | Δφ_true vs B (mean, paired) | frac. ticks improved |
|---|---|---|
| A (reference_nullspace) | −0.0000 | 29% |
| C, λ=1 | −0.0001 | 11% |
| C, λ=3 | −0.0002 | 13% |
| C, λ=10 | −0.0005 | 13% |
| C, λ=30 | −0.0015 | 17% |

**Cost incurred for that non-benefit** (E_chosen = predicted-vs-desired beam
position error at the given horizon step, mm):

| mode | E_chosen +10 (mm) | joint effort (frac. of velocity limit) |
|---|---|---|
| B (none) | 0.204 | 0.111 |
| C, λ=1 | 0.205 | 0.111 |
| C, λ=3 | 0.211 | 0.120 |
| C, λ=10 | 0.249 | 0.194 |
| C, λ=30 | 0.387 (+90% vs B) | 0.292 (+163% vs B) |

**Root cause, isolated via the directional-derivative decomposition**
(`ΔJ_component = f_component·d`, `d` = realized command change, at idx=70
ordinary and idx=127 the global φ-maximum): the QP mechanics are correct —
`f_manip` dominates the linear cost and reliably decreases the *surrogate* in
the chosen direction at every λ. The failure is that **the linear surrogate
itself mispredicts the sign of the true φ change in 40–43% of ticks**,
uniformly across all four λ — not concentrated only at the φ-peak tail
identified in the earlier pilot. Increasing λ just makes the optimizer chase
a locally-plausible but globally-unreliable model harder; it cannot fix a
directional-fidelity problem.

## Gradient-staleness diagnostic — hypothesis tested and rejected

Natural follow-up question: is the surrogate failing because it's linearized
at `z_ref` while the QP acts on off-reference `z_live` states ("gradient
staleness")? Directly tested: `cos∠(g_N(z_ref), g_N(z_live))` at 10 wrong-sign
and 10 correct-sign ticks, using the same state-dependent beam-Jacobian
provider as Phase 0.

| | cos∠(g_ref, g_live) |
|---|---|
| wrong-sign ticks | +0.996 ± 0.005 |
| correct-sign ticks | +0.987 ± 0.015 |

**Rejected.** The gradient direction barely rotates between `z_ref` and
`z_live` in either group — the controller tracks closely enough that this
isn't the mechanism. The failure is not *where* the gradient is evaluated;
it is that a first-order extrapolation from an accurate gradient still
overshoots the true (concave, diminishing-return-by-construction) φ surface
over the step size the QP actually takes — a curvature/step-size failure,
not a staleness failure. This is consistent with φ's own definition
(`log(1+σ²)`, derivative → 0 as σ grows or as φ nears a local maximum, by
design) and with the earlier pilot's finding that the worst sign-disagreement
concentrated right at the path's global φ-maximum.

## Why A ≈ B here (and why that does NOT mean `Q_N` is unnecessary)

The replay found A (reference_nullspace) and B (none) nearly indistinguishable
in instantaneous command, tracking, and φ. This is a property of the replay
design, not evidence against `Q_N`: the replay states come from the validated
controller, which — because it already ran with `Q_N` active — stayed close
to the reference nullspace posture throughout. A single-tick counterfactual
replay is structurally unable to reproduce the accumulated redundant-state
drift that `Q_N` suppresses over a full closed-loop run (see the horizon/
posture analysis elsewhere in this package). `Q_N` remains the defensible
choice: it doesn't directly improve the primary task, but it stabilizes
redundant posture without materially interfering with tracking, whereas the
tested manipulability alternative actively degrades both tracking and effort
without the authority benefit it was meant to provide.

## Disposition

Closed as a validated negative result, not unfinished work. A plausible next
step (`g_N` evaluated at predicted/current state with sequential
relinearization and a nullspace trust region — closer to an SQP layer) could
address the curvature/step-size failure directly, but requires repeated
state-dependent beam-Jacobian evaluations off the reference manifold, which
are expensive and were explicitly deferred (this was never meant to be
solved before this paper). Infrastructure (`authority_nullspace_mpc.py` and
everything in `scripts/`) is preserved for that future work.
