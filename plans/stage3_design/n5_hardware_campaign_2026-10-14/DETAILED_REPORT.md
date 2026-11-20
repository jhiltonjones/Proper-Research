# Three-Condition Hardware Campaign (n=5 per condition) — Detailed Report

**Date:** 2026-10-14/15. **Plan:** `plans/stage3_design/n5_hardware_campaign_2026-10-14/` (this directory). **Reproducibility:** every number below is produced by `scripts/*.py` reading the raw run directories named in `scripts/run_registry.py`; full machine-readable output is in `results/*.json`.

## 1. Context and methodology

Three conditions, 5 hardware repetitions each, same floor240 vessel plan, same triangle path:

- **NEW** — contact MPC with the smooth, state-conditioned J1/J3 Jacobian hybrid.
- **OLD-C** — contact MPC with the original scheduled contact Jacobian J1 only.
- **NC** — MPC with the scheduled no-contact Jacobian (contact not modelled at all).

This supersedes an earlier n=3 attempt that was invalidated by a timing regression in NEW (`project_s_q` costing 25-45ms/tick from an unvectorized 2001-candidate search; fixed with a 229x speedup, bit-exact validated offline — see `speed_fix_validation/`). All three pairwise comparisons (NEW-vs-OLD-C, OLD-C-vs-NC, NEW-vs-NC) are kept explicitly distinct throughout; they are never collapsed into one ranking.

Region boundaries were fixed **before** inspecting these 15 runs, from the prior investigation: R_A ≈ 12-17mm (shared early low-authority zone), R_C = 55-65mm (the established late contact-sensitive region). A ±2-3mm sensitivity check on R_C is reported where relevant.

---

## 2. Evidence: timing/solver practical-equivalence gate

**Why this gate exists:** the entire comparison is invalid if NEW's control loop still runs on a different real-time cadence than OLD-C's. Tolerances were fixed before looking at these runs: tick_dt and solver_time within 10ms, realization ratio within 0.05, time-limited-QP fraction within 5 percentage points.

| metric | NEW (median) | OLD-C (median) | \|diff\| | tolerance | result |
|---|---:|---:|---:|---:|---|
| tick_dt (ms) | 110.80 | 110.30 | 0.50 | 10.0 | PASS |
| solver_time (ms) | 9.57 | 9.02 | 0.55 | 10.0 | PASS |
| realization ratio | 0.938 | 0.957 | 0.019 | 0.05 | PASS |
| % QP hit time-limit | 0.2% | 0.4% | -0.2pp | 5pp | PASS |
| deadline misses | 0 / 2470 ticks | 0 / 2483 ticks | — | 0 required | PASS |

Full per-run breakdown (`results/control_step_characterization.json`):

| cond | rep | control steps | tick_dt (ms) | solver (ms) | realization | %solved | %inaccurate | %time-limit |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NEW | 1 | 495 | 110.6 | 9.33 | 0.933 | 99.8 | 0.2 | 0.0 |
| NEW | 2 | 494 | 110.9 | 9.70 | 0.944 | 99.8 | 0.0 | 0.2 |
| NEW | 3 | 495 | 110.8 | 9.57 | 0.938 | 99.6 | 0.0 | 0.4 |
| NEW | 4 | 494 | 110.9 | 9.60 | 0.931 | 98.6 | 1.2 | 0.2 |
| NEW | 5 | 494 | 110.6 | 9.44 | 0.942 | 99.0 | 0.6 | 0.4 |
| OLD-C | 1 | 497 | 110.3 | 9.02 | 0.971 | 97.4 | 1.8 | 0.8 |
| OLD-C | 2 | 497 | 110.2 | 8.93 | 0.957 | 98.6 | 1.0 | 0.4 |
| OLD-C | 3 | 498 | 110.0 | 8.76 | 0.960 | 98.4 | 1.2 | 0.4 |
| OLD-C | 4 | 496 | 110.5 | 9.23 | 0.945 | 98.2 | 1.4 | 0.4 |
| OLD-C | 5 | 495 | 110.6 | 9.42 | 0.939 | 97.8 | 1.2 | 1.0 |
| NC | 1 | 389 | 111.7 | 10.49 | 0.933 | 88.9 | 2.1 | 9.0 |
| NC | 2 | 389 | 112.1 | 10.84 | 0.915 | 87.9 | 1.8 | 10.3 |
| NC | 3 | 387 | 112.3 | 11.05 | 0.912 | 89.1 | 2.1 | 8.8 |
| NC | 4 | 390 | 111.7 | 10.54 | 0.924 | 87.7 | 3.8 | 8.5 |
| NC | 5 | 387 | 112.3 | 11.10 | 0.915 | 89.4 | 2.3 | 8.3 |

**Observation (not part of the gate, flagged for completeness):** NC hits the 30ms solver time-limit far more often (~8.3-10.3% of ticks) than either contact condition. This doesn't affect the NEW-vs-OLD-C gate (which is what matters for the primary causal comparison) and all runs still completed with zero deadline misses.

**Limitation, stated rather than papered over:** the new in-QP TCP-workspace constraint's configuration (bounds, margin, live-relinearize setting) is not recorded in `controller_metadata.json` — confirmed by direct inspection, a gap in the logging work this round, not fixed retroactively. "Configured identically across conditions" rests on knowing the exact commands used (none passed `--disable-tcp-workspace-in-qp` or a custom margin), not a direct artifact check.

**Conclusion:** Gate **PASSES**. The timing regression that invalidated the n=3 round is gone. Everything below can be interpreted causally for the NEW-vs-OLD-C comparison specifically.

---

## 3. Evidence: primary physical outcome

### 3.1 Whole-run (every tick, no progress clipping)

| | NEW | OLD-C | NC |
|---|---|---|---|
| RMSE (mm), 5 reps | 0.642, 0.668, 0.658, 0.691, 0.681 | 0.741, 0.720, 0.705, 0.731, 0.717 | 1.421, 1.447, 1.491, 1.494, 1.426 |
| RMSE median | **0.668** | 0.720 | 1.447 |
| RMSE range | [0.642, 0.691] | [0.705, 0.741] | [1.421, 1.494] |
| median error (mm), 5 reps | 0.385, 0.383, 0.385, 0.425, 0.393 | 0.466, 0.417, 0.389, 0.433, 0.393 | 0.467, 0.476, 0.461, 0.461, 0.477 |
| median error, median-of-medians | **0.393** | 0.417 | 0.467 |
| mean lag (mm) | -0.33 | -0.30 | +0.27 |
| mean cross-track (mm) | 0.29 | 0.31 | 0.37 |

NEW's whole-run RMSE **fully range-separates** below OLD-C's (NEW's worst rep, 0.691mm, still beats OLD-C's best, 0.705mm) — the strongest clean result in this campaign. Median error shows the same direction, smaller margin.

### 3.2 Region C = [55, 65]mm (the established contact-sensitive region)

| | NEW | OLD-C | NC |
|---|---|---|---|
| RMSE (mm), 5 reps | 0.549, 0.588, 0.662, 0.565, 0.588 | 0.648, 0.607, 0.618, 0.726, 0.681 | 0.461, 0.484, 0.606, 0.497, 0.500 |
| RMSE median | **0.588** | 0.648 | 0.497 |
| RMSE range | [0.549, 0.662] | [0.607, 0.726] | [0.461, 0.606] |
| median error (mm), 5 reps | 0.282, 0.313, 0.286, 0.307, 0.289 | 0.350, 0.347, 0.334, 0.327, 0.325 | 0.293, 0.323, 0.294, 0.362, 0.379 |
| median error, median-of-medians | **0.289** | 0.334 | 0.323 |
| max error (mm), median of 5 | 1.97 | 2.10 | 1.06 |
| mean lag (mm) | -0.60 | -0.55 | -0.19 |

**Primary effect, defined precisely (no rep-number pairing assumed):**

```
Delta_C = median_r(RMSE_NEW,C) - median_r(RMSE_OLD-C,C) = 0.588 - 0.648 = -0.060mm
```

Run-level bootstrap 95% CI (resampling the 5 vs 5 runs, **never ticks**, n=10000): **[-0.139, +0.015]mm**.

**±2-3mm region-C boundary sensitivity** (reported alongside, not instead of, the primary [55,65]mm boundary): the NEW<OLD-C direction holds under [53,67]mm and [52,68]mm as well (NEW RMSE medians 0.556mm and 0.552mm vs OLD-C's 0.628mm and 0.605mm) — the result is not an artifact of the exact boundary chosen.

### 3.3 Region A = [12, 17]mm (shared low-authority zone, specificity check)

| | NEW | OLD-C |
|---|---|---|
| RMSE median | 1.23* | 1.65–2.02* |
| (full per-rep breakdown in `results/phase1_2_results.json`) | | |

*(figures from the equivalent n=3-round region-A finding, re-confirmed in direction at n=5 — see `results/phase1_2_results.json` for this round's exact per-rep region-A numbers.)*

**Conclusion (Section 3):** Delta_C is negative (NEW better) and the direction is robust to the region-C boundary choice, but the bootstrap CI includes zero — **not statistically significant at n=5**, though consistently favorable. The whole-run comparison is a cleaner, fully-separating result. NEW's advantage is **not confined to region C** (it also shows in region A) — this doesn't falsify the mechanistic hypothesis (only a region-A *worsening* would), but it means the benefit isn't narrowly localized to the contact-sensitive zone the mechanism specifically predicts.

---

## 4. Evidence: how NEW actually used the hybrid map

Every tick of NEW's 5 reps was classified into pure-J3 / blend / schedule-fallback (r_q ≥ r_b, map not trusted) / map-invalid, using the production `ContactJacobianHybridModel` (memorylessness re-verified bit-exact at n=5: forward vs. reversed replay order gives identical `s_q`/`r_q`/`w`, max abs diff = 0.0).

| rep | blend | pure-J3 | schedule-fallback | fallback s_q range |
|---|---:|---:|---:|---|
| 1 | 49.1% | 0.8% | 50.1% | [0.79, 53.81]mm |
| 2 | 48.8% | 0.8% | 50.4% | [0.68, 53.63]mm |
| 3 | 52.1% | 1.0% | 46.9% | [0.68, 53.23]mm |
| 4 | 55.3% | 1.0% | 43.7% | [1.05, 52.09]mm |
| 5 | 53.2% | 1.0% | 45.7% | [0.82, 53.23]mm |

**Conclusion:** across all 5 reps, NEW falls back to the plain J1 schedule (not using the hybrid map at all) for roughly **44-55% of ticks**, concentrated in s ≈ 0.7-54mm — i.e. essentially the whole early-to-mid path, transitioning into blend mode right around where region C (55-65mm) begins. **This means NEW's whole-run advantage (Section 3.1) cannot be mechanistically attributed to the hybrid map** for most of the path, since NEW is running identically to OLD-C's Jacobian there. Only the region-C-specific comparison is plausibly attributable to the map at all.

---

## 5. Evidence: model fidelity (E_J,D) vs the model-oracle J4

`E_J,D = ||(J4 - J_used) @ diag(D)||_F`, where J4 is the full contact-model Jacobian recomputed at the measured hardware state (an offline model-oracle, confirmed in this investigation's own earlier rounds to be a model oracle, not independent physical ground truth).

**Pooled whole-run mean E_J,D** (all sampled ticks, all 5 reps): NEW = 0.00098, OLD-C = 0.00109, NC = 0.00151. NEW < OLD-C in this coarse view.

**Condition-level, region-C mean (5mm-binned, averaged across 5 reps):** `Delta(E_J,D)_NEW-OLD-C = -0.00031` (NEW lower — i.e. better).

This condition-level number is **confounded**: it compares mean curves built from each condition's own realized states, which differ between conditions. Section 6.2 gives the properly state-paired version.

---

## 6. Evidence: same-state MPC counterfactual (the central causal test)

At sampled anchor states (full region-C coverage + stride-5 elsewhere, 153-157 anchors per run), the controller is re-solved with the *same* measured state/reference/delay/constraints, varying only the Jacobian source: NEW's own hybrid, OLD-C's plain J1, NC's own schedule, and the model-oracle J4 (computed at the actual measured state for "now", and at the reference's own future states for the horizon). Reproduction of each condition's own real logged command was verified first (mean cosine similarity 0.84-0.91 across all 15 runs on meaningful-magnitude ticks — consistent with this investigation's established baseline).

### 6.1 Condition-level (unpaired — each condition's own states)

| branch | mean θ_u vs J4 (deg) |
|---|---:|
| NEW, own | lower (hybrid closer to oracle command direction) |
| OLD-C, own (=J1) | higher |
| region-C mean Δθ_u, NEW−OLD-C | **-1.12°** (NEW better) |

### 6.2 Within-anchor, state-paired (the rigorous version — same exact state, only J changes)

```
Delta_E_J(x_k)     = E_J,D(J1 branch)     - E_J,D(own branch)      [positive = NEW better]
Delta_theta_u(x_k) = theta_u(J1 branch)   - theta_u(own branch)    [positive = NEW better]
```
computed at NEW's own 357 region-C anchors (pooled across its 5 reps):

| | n | median | fraction positive (NEW better) | range |
|---|---:|---:|---:|---|
| Delta_E_J | 357 | **-0.00012** | **0.050** | [-0.00144, +0.00009] |
| Delta_theta_u | 357 | +0.382° | 0.619 | [-113.7°, +37.7°] |

**This is the key evidence in the whole campaign.** At **95% of NEW's region-C anchor states**, NEW's own Jacobian is actually *slightly farther* from the model-oracle J4 than the plain J1 schedule would have been at that exact same state. Command angle is more mixed (62% of anchors favor NEW, small positive median) but with an enormous spread, including one anchor where NEW's command was 113.7° *worse* in direction than J1's would have been.

**Conclusion:** the condition-level comparison (6.1) and the state-paired comparison (6.2) **disagree**. The condition-level number is confounded by *which* states each condition visited; once state is held fixed, the "better local Jacobian" story mostly disappears for E_J,D specifically, and is weak/mixed for command angle. This is exactly the kind of confound the within-anchor design was built to catch.

---

## 7. Evidence: chronology (does model fidelity actually predict later physical consequence?)

`E_future(k) = (1/H) * sum_{h=d}^{d+H-1} ||e_{k+h}||` using the real delay d=2 and horizon H=15, linking a tick's model fidelity to its own run's own later physical error (never progress-matched across runs).

**Condition-aggregate, region C:**

| | Δ(NEW−OLD-C) | direction |
|---|---:|---|
| E_J,D | -0.00031 | down ✓ |
| θ_u | -1.122° | down ✓ |
| E_future | -0.066mm | down ✓ |

**Chronology holds in full** (E_J,D down → θ_u down → E_future down, in that order) **at the condition-aggregate level**. Per-run within-run correlation between E_J,D(k) and E_future(k) is weak and inconsistent in sign (NEW: -0.09 to +0.38 across its 5 reps; OLD-C: +0.08 to +0.32; NC: consistently +0.56 to +0.63) — reported descriptively only, given temporal autocorrelation.

**Conclusion:** the aggregate chronology is internally consistent and looks like a clean causal chain — **but this is the same aggregate/condition-level view that Section 6.2 showed is confounded by state selection.** It should not be read as independent confirmation of the mechanism; it's a restatement of the same confounded comparison across one more link in the chain.

---

## 8. Evidence: configuration departure (r_q) — the proposed virtuous-cycle test

`r_q` = dimensionless, D_Q-normalized, tangent-projected off-path distance, computed identically for all three conditions (generalized from NEW's own hybrid model, confirmed bit-exact against the production `ContactJacobianHybridModel.r_q`, max abs diff ~1e-12).

| region | NEW median r_q | OLD-C median r_q | NC median r_q |
|---|---:|---:|---:|
| A [12,17]mm | 13.09 | 11.78 | 11.81 |
| C [55,65]mm | **8.61** | 9.08 | 9.26 |

`r_q^NEW < r_q^OLD-C` holds in region C (8.61 < 9.08) — **descriptive support** for "better command → stays closer to the planned configuration → the local model stays valid longer." Per this plan's own pre-registered rule, this is **only interpreted as supporting the mechanism if Section 7's chronology also holds**. It does (Section 7) — but Section 7 is itself the confounded aggregate view, and Section 6.2's rigorous test does not support the underlying premise ("better command" at the Jacobian level). So this finding is reported as present, consistent with the aggregate story, but **not independent confirmation** given the tension in Section 6.

---

## 9. Evidence: safety and command smoothness

**Absolute feasibility (hard gate, checked first):** `h_j > 0` for every hard constraint face, every run, every tick. **PASS, zero violations, all 15 runs.**

| | NEW min margin | OLD-C min margin | NC min margin | closest face |
|---|---:|---:|---:|---|
| median across 5 reps | +1.76mm | +1.72mm | +1.77mm | z_tight_max (all) |

(Wide Cartesian workspace box margin not numerically reconstructable from logs — evidenced instead by zero `tcp_out_of_workspace` aborts across all 15 runs, same limitation as documented in the n=3 round.)

**Command smoothness** (`||u_k - u_{k-1}||`, joint velocities):

| | NEW median p99 | OLD-C median p99 | NC median p99 | NEW median max | OLD-C median max | NC median max |
|---|---:|---:|---:|---:|---:|---:|
| | 0.062 | 0.061 | 0.039 | 0.096 | 0.080 | 0.059 |

**Conclusion:** no safety degradation. Command smoothness is essentially the same between NEW and OLD-C (p99 within 2%); both contact conditions are rougher than NC (expected — contact disturbance vs. none). **NEW does not purchase its tracking advantage at the cost of safety margin or smoothness.**

---

## 10. Three pairwise comparisons, kept distinct

| comparison | question | result |
|---|---|---|
| **(a) NEW vs OLD-C** (state-conditioning value) | Does measured-state conditioning improve on the nominal schedule? | Delta_C = -0.060mm, right direction, CI includes zero. Whole-run fully separates. Mechanistically unsupported at the rigorous within-anchor level. |
| **(b) OLD-C vs NC** (contact-modelling value) | Does explicit contact modelling matter at all? | OLD-C's own region-C RMSE median (0.648mm) is *worse* than NC's (0.497mm) in OLD-C's own contact-sensitive region — contact disturbance measurably hurts the plain scheduled-contact approach relative to not modelling contact. |
| **(c) NEW vs NC** (end-to-end gain) | What does contact-aware + state-conditioned modelling buy over ignoring contact entirely? | NEW's region-C RMSE median (0.588mm) is still +0.090mm *worse* than NC's (0.497mm) — the gap to "don't model contact" has not been closed. |

These are **not** collapsed into a ranking. (b) and (c) together are a genuinely interesting, separate finding: NC's apparent region-C performance edge over both contact conditions is plausibly because NC's plan/path in this region simply never engages the wall the way the contact plans do — this is flagged as worth a dedicated look, not resolved here.

---

## 11. Preregistered hypothesis checklist (summary)

| hypothesis | result |
|---|---|
| Timing/solver gate | **PASS** |
| Primary (Delta_C < 0) | **PASS** (point estimate), CI includes zero — not significant at n=5 |
| Mechanistic — within-anchor (primary test) | **NOT MET** (E_J,D worse at 95% of anchors) |
| Mechanistic — condition-level (secondary) | PASS, but confounded (see Section 6) |
| Specificity (no advantage required in region A) | NEW *did* improve region A too — doesn't falsify, but benefit not localized |
| Safety | **PASS**, zero violations |
| Chronology | Holds in full at the aggregate level; not independent evidence given the within-anchor tension |

**Four-outcome classification (from the pre-registered decision tree):**

> *"Within-anchor E_J,D/θ_u NOT consistently down → the hybrid did not deliver its intended model-level advantage on these anchors."*

---

## 12. Overall conclusion

The physical tracking improvement for NEW is **real in direction** — consistent across the whole-run comparison (fully separating), the region-C median difference, and robust to the region-C boundary choice — but **not yet statistically significant at n=5** (bootstrap CI for the region-C effect includes zero).

The **mechanistic explanation is not established**. Every *aggregate/condition-level* metric (E_J,D, θ_u, E_future, r_q) tells a clean, internally consistent story where better local Jacobian fidelity causes the improvement. But the single most rigorous test available — holding the exact measured state fixed and varying only the Jacobian source — does not support that story: NEW's hybrid Jacobian is actually slightly *worse* than the plain schedule at matching the model-oracle at 95% of the states that matter (region C). The aggregate-level agreement is most plausibly an artifact of comparing different realized trajectories rather than genuine evidence of the proposed causal chain.

**What this means practically:** NEW's hardware behavior is modestly better than OLD-C's, and that improvement costs nothing in safety or smoothness — but *why* it is better is still an open question. It is not confidently "because the hybrid Jacobian is more accurate." Plausible alternative explanations (not yet tested): the hybrid's smoothstep blending could be providing a regularizing/damping effect on the command independent of raw Jacobian accuracy; the effect could be smaller than it looks and partly driven by ordinary run-to-run variability; or the real mechanism could lie somewhere the current oracle/anchor design doesn't capture.

**Recommended next step:** either (a) more hardware repetitions to tighten the bootstrap CI and the within-anchor fraction-positive estimate, or (b) a direct investigation of why the within-anchor and condition-level tests diverge so sharply (e.g. checking whether a small number of outlier anchors dominate the aggregate curves) — neither has been done yet.
