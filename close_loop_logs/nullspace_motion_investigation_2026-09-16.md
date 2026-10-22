# Null-space motion investigation — S-curve, bimaterial model, 2026-09-16

## 1. Motivating question

The MPC controllers (`mpc_lti`, `mpc_ltv_offline`, `mpc_ltv_sqp_online`) command joint/insertion
velocities `v` for a kinematically redundant system (6 joints + insertion, 3 task outputs), so
`v` can contain motion in `Null(J)` that does not move the beam tip. Two hypotheses were raised
for why this null-space content might be large enough to matter:

1. The `(v - v_ref)^T R (v - v_ref)` input-tracking term pulls the solution toward `v_ref`, and
   `v_ref` itself has null-space content (from the offline planner) — so the MPC could be
   reproducing/inheriting that content rather than generating its own.
2. The null-space component the QP settles on could be genuinely large and control-theoretically
   wasteful (a candidate justification for adding a soft null-space penalty to the cost).

A formal 9-experiment protocol was defined (Experiments 1–9, only 1–7 executed; 8 is gated on
1–7 showing necessity, 9 is conceptual/no test) to test this rigorously rather than relying on
an informal null-fraction comparison. All experiments below ran on the **S-curve (10×20mm,
bimaterial offline planner)** only, same-day, single-rep round-robin interleaving across
conditions (to control for lighting/camera drift across a run), horizon=15,
`state_tracking_weight=0.001` fix active — the same base MPC parameters used throughout this
project's MPC-vs-inverse-Jacobian work. n=3 reps per condition unless noted.

**Read this document together with §5 — the central caveat that motivated writing it up now.**

## 2. Baseline: inv / LTI / LTV on the S-curve (Experiments 1–2, interleaved, n=3)

| Controller | RMS (mm) | Max (mm) | Hold (mm) | Final (mm) |
|---|---|---|---|---|
| Inverse-Jacobian+FF | 1.770±0.053 | 4.258±0.199 | 0.285±0.068 | 0.155±0.074 |
| MPC-LTI+FF | **1.671±0.051** | **3.933±0.108** | 0.377±0.087 | 0.330±0.292 |
| MPC-LTV+FF | 1.787±0.032 | 4.307±0.039 | 0.290±0.087 | 0.279±0.113 |

Mixed result on this shape: LTI wins RMS/max, LTV wins hold, inverse-Jacobian wins final — unlike
the cleaner "MPC beats inv on every metric" pattern seen on rectangle/triangle/U-shape
(`mpc_vs_inv_jacobian_2026-09-15.md`). One genuinely clean sub-comparison falls out of this,
though: at the time these three ran, `inv` and `lti` (unknown to us — see §5) both used the exact
same **frozen** analytical Jacobian source. So `inv` vs `lti` here is an apples-to-apples "does the
QP-MPC formulation beat plain damped-least-squares, holding the Jacobian source fixed" test, and
MPC-LTI wins RMS/max under that fixed comparison.

**Null-space projector norms** (`ρ_N`-style decomposition, `P_N = V_n V_n^T` from the SVD of each
controller's own local Jacobian, n=3 averaged):

| Controller | mean ‖v_ref,N‖ | mean ‖v_MPC,N‖ |
|---|---|---|
| LTI | 0.0095±0.0000 | 0.0839±0.0009 |
| LTV | 0.0026±0.0000 | **0.0518±0.0007** |
| "tf" (as run here — see §5) | 0.0026±0.0000 | 0.0966±0.0012 |

LTV's solved commands carry noticeably less null-space content than LTI's, despite LTV *not*
beating LTI on RMS/max — i.e. less null-space motion does not straightforwardly track with better
tip tracking on this shape, consistent with §4's Experiment 7 finding that null motion is not
simply wasted.

## 3. Cost-structure experiments (3, 4, 5) — clean, unconfounded by §5's issue

These vary the MPC's own cost terms on the (fixed-schedule) LTV controller; none of them touch
live relinearisation, so they are not affected by the caveat in §5.

### Experiment 3 — is `v_ref` in the R-term pulling in null-space content?
Diagnostic: `J_R = v^T R v` (input cost with the reference target zeroed) vs the normal
`(v - v_ref)^T R (v - v_ref)`, everything else (Q, Qp, Rd) unchanged.

| Condition | RMS (mm) | Max (mm) | Hold (mm) | Final (mm) |
|---|---|---|---|---|
| Normal (v_ref present in R) | 1.800±0.045 | 4.286±0.145 | 0.350±0.039 | 0.377±0.163 |
| v_ref zeroed in R | 1.797±0.035 | 4.256±0.134 | 0.352±0.053 | 0.332±0.110 |

**No effect** (differences are well inside the noise floor set by the two conditions' own std).
Hypothesis 1 from §1 is **rejected**: the R-term's pull toward `v_ref` is not a driver of excess
null-space motion.

### Experiment 4 — is the input-increment smoothing term (`Rd`) suppressing useful correction?
`R_d = 0` (increment-smoothing penalty removed) vs normal.

| Condition | RMS (mm) | Max (mm) | Hold (mm) | Final (mm) |
|---|---|---|---|---|
| Normal | 1.800±0.045 | 4.286±0.145 | 0.350±0.039 | 0.377±0.163 |
| `Rd=0` | 1.814±0.034 | 4.253±0.028 | 0.359±0.056 | 0.351±0.159 |

**No effect.** `Rd` smoothing is not a dominant driver.

### Experiment 5 — DARE terminal cost
DARE terminal cost disabled vs normal (normal uses it, `use_dare_terminal_cost=True`).

| Condition | RMS (mm) | Max (mm) | Hold (mm) | Final (mm) |
|---|---|---|---|---|
| Normal (DARE on) | 1.800±0.045 | 4.286±0.145 | 0.350±0.039 | 0.377±0.163 |
| DARE disabled | 1.867±0.010 | 4.236±0.089 | **0.313±0.055** | **0.195±0.034** |

Small but real, non-overlapping effect, and it's a **trade-off, not a one-sided win**: disabling
the terminal cost slightly *worsens* transit-phase RMS (+3.7%) while meaningfully *improving*
hold/final tracking (−11% hold, −48% final). Consistent with what a terminal cost should do —
buy transit-phase stability at some cost to how tightly the endpoint gets nailed down. Not a major
driver of excess null-space motion either way; not a slam-dunk case for turning it off.

### Experiment 6 — R/Rd row/null coupling
Resolved analytically, no live-robot cost. `R = input_tracking_weight · diag(1/input_error_scale²)`
and `Rd` are built from `input_error_scale = (0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 5.0e-3)` and
`acceleration_limit` — uniform across all six joint entries. A uniform (scalar-times-identity)
weight in the 6-joint block is invariant under any orthogonal change of basis, so **by
construction** the row-space/null-space cross-coupling block `R_nr = 0` exactly, for any
Jacobian. No coupling to test for.

## 4. Horizon-usefulness (Experiment 7)

Tests whether "instantaneously null" motion is actually useful over the prediction horizon,
using `γ_{H,k} = ‖G_{0,k} v_{0|k}‖ / (‖G_{0,k}‖‖v_{0|k}‖ + ε)` where `G_{0,k}` stacks
`Qp^{1/2} J_{k+j|k}` over the horizon. The "problematic" case the experiment is designed to catch
is high instantaneous null-fraction **and** low `γ_H` (large null motion that's also useless over
the horizon — the case that would justify Experiment 8's soft null-space penalty).

**Result:** 0 of 156 ticks analysed fell into the problematic quadrant. Even the ticks with the
highest instantaneous null-fraction retained substantial horizon usefulness (mean `γ_H = 0.327`
on high-null-fraction ticks vs `0.488` overall — reduced, but far from zero). This measurement
was made in-session against the LTV run's logged horizon predictions; no standalone script for it
was kept, so it is reported as an in-session finding rather than something with a file path to
re-run.

**Consequence:** Experiment 8 (soft null-space penalty `ρ_N‖P_N v‖²`, swept and evaluated on
RMSE/max/‖v‖/‖Δv‖/constraint utilization) was **not run** — its own gating condition was not met.
Adding a null-space penalty is not currently justified by this data.

## 5. The confound this write-up needs to flag prominently: the online Jacobian is not actually being computed live

Experiments 1 and 2 (in their original form) were meant to compare a Jacobian **frozen at t=0**
(LTI), a Jacobian **precomputed once per reference sample and never revisited** (LTV), and a
Jacobian **relinearised at the measured state every control tick** ("tick-frozen",
`mpc_ltv_sqp_online`) — isolating whether *staleness* or *intra-horizon variation* matters.

Investigating §2's counter-intuitive null-space numbers turned up a real bug:
`AnalyticalBeamJacobianProvider.__call__(self, state)` **ignores its `state` argument** and
always returns the Jacobian computed once at construction — a deliberate, documented tradeoff
("computed once, frozen — a local linearisation... fine for a small translation-only move"), not
something that had been surfacing as a caveat. Both `inv` (damped least squares) and, in its
original form, "tick-frozen" used this same frozen provider by default. So the "tf" row in §2's
table is not a live-relinearisation condition at all — it's a second frozen-Jacobian run under a
different label, and **Experiment 2's original conclusion (tick-frozen ≈ LTI, both ≠ LTV) is not
evidence about live relinearisation.** It only shows two frozen-Jacobian conditions resembling
each other, which is expected.

Fixing this (swapping in `beam_jacobian_providers.from_model_bundle`, a genuinely state-dependent
provider, verified self-consistent against its own finite-difference check) and re-running
live surfaced a second, independent problem: the provider's per-call cost is **not** constant.
Along the smooth, precomputed reference manifold (as LTV's schedule-building uses it) it costs
~9ms/call; evaluated fresh at genuinely measured, off-manifold live states it costs **188–494ms
per call** (mean 254ms) — the underlying nonlinear beam-equilibrium solve needs far more Newton
iterations to converge off the smooth manifold. Relinearising every tick therefore collapsed the
real control loop from the nominal 10Hz to ~3.2Hz (confirmed: `inv`/`lti`/`ltv` all hold a rock
steady 0.100s/tick; only the genuinely-live "tf" showed 0.194–0.591s/tick), which is slow enough
to intermittently starve the robot of continuous motion (measured `|Δz_actual|/|Δt·v_commanded|
≈ 0.01`) — producing a 9.46mm-RMS divergent live run that is a **real-time performance artifact,
not a control-theoretic one**. Confirmed by offline closed-loop simulation against the nonlinear
plant (no wall-clock delay modeled): relinearising every tick there gives RMS=0.115mm, matching
LTV (0.120mm) and beating LTI (0.251mm) — the control law itself is sound.

A throttled fix (`relinearise_every=N`, reuse the last Jacobian for `N-1` ticks) was added and
validated offline: RMS stayed flat at 0.115mm from `N=1` to `N=20` in the idealized simulation. A
live smoke test at `N=5` avoided the divergence (RMS 3.58mm, max 6.37mm, `path_complete`, median
inter-tick period recovered to 0.092s) but did not reach the ~1.7mm baseline either — throttling
made tick timing *bursty* (slow ~0.3–0.67s relinearising ticks alternating with fast 5–20ms cached
ticks), and the `progress_mode="wallclock"` reference-pacer reacted to that burstiness with sudden
multi-sample target jumps (`corr(dt, Δref_index) ≈ −0.09`, e.g. +1 sample after a 0.59s tick then
+5 samples on the next 16ms tick) that the rate-limited plant cannot track instantaneously. This
is a distinct, second, still-open issue.

**Bottom line for this section, stated plainly:** *the live/online computation of the beam
Jacobian is not currently being evaluated correctly inside the real-time control loop — either
it's silently frozen (the original bug) or it's slow enough to break the timing budget it runs in
(the fix), and even after partially throttling it, a second timing-related artifact (bursty
progress-pacing) remains unresolved.* This is a real-time engineering problem, not a property of
the control law, and it has **not** been fixed. No conclusion in this document should be read as
"live relinearisation helps" or "live relinearisation hurts" — that question remains genuinely
open, gated on a real-time-engineering fix that has not been made. Wiring
(`relinearise_every`) exists end-to-end from `mpc_variants.build_sqp_online_mpc` through
`close_loop_path_follow.py`/`compare_controllers_live.py`'s `--mpc-relinearise-every` flag, but
should be treated as a partial mitigation under active investigation, not a validated setting.

Experiment 1 (inverse-Jacobian null-space baseline) inherits this same problem: its intended
methodology assumed `inv` relinearises live, which it does not (it uses the same frozen
provider). Experiment 1's original run is therefore not a meaningful "live relinearisation"
baseline either, and needs to be re-specified once §5's real-time issue is resolved.

## 6. What's actually settled vs. what's still open

**Settled (clean, unconfounded by §5):**
- The R-term's pull toward `v_ref` is not driving excess null-space motion (Exp 3).
- `Rd` input-increment smoothing is not a dominant driver (Exp 4).
- The DARE terminal cost trades transit RMS against hold/final accuracy; it is not a major driver
  of excess null-space motion in either direction (Exp 5).
- `R`/`Rd` have zero row/null cross-coupling by construction, for any Jacobian (Exp 6).
- No tick in 156 analysed showed both high instantaneous null-fraction and low horizon usefulness
  (Exp 7) — a soft null-space penalty (Exp 8) is not currently justified and was not run.
- Holding the Jacobian source fixed, MPC-LTI's QP formulation beats plain damped-least-squares
  inverse-Jacobian on RMS/max (§2's inv-vs-lti sub-comparison).

**Still open, explicitly not settled by this investigation:**
- Whether genuine live (measured-state) relinearisation helps or hurts tracking, control-
  theoretically — blocked on fixing the real-time cost of the state-dependent Jacobian provider
  and the bursty-timing/wallclock-pacing interaction it exposed.
- Experiment 1's inverse-Jacobian null-space baseline, correctly specified.
- Experiment 8 (soft null-space penalty), gated on Exp 7 showing necessity — it didn't.
- Experiment 9 (selective singular-value damping) — conceptual only, not tested, and explicitly
  flagged as not equivalent to exact null-space suppression regardless.

Given §6's settled results, the original motivating hypotheses (§1) are not supported: neither
the R-term's `v_ref` pull nor the increment-smoothing term explain the null-space motion observed
in this MPC, and the horizon-usefulness analysis suggests that motion is not being wasted in the
first place. The honest overall conclusion is that this protocol did not find a cost-structure
cause for excess null-space motion on the S-curve — and, separately, exposed a real-time
engineering gap in how the "live relinearisation" condition has been implemented, which is the
actual open item, not a cost-tuning question.
