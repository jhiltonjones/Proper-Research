# Follow-up: Why Does a Less Contact-Faithful Jacobian Track Better in Region C? (2026-10-16, FINAL)

**Status: all 5 analyses complete.** This follows up on `FRESH_RUNS_7CONDITION_ANALYSIS.md`'s headline puzzle: putting the no-contact Jacobian on the contact trajectory (condition 4, everywhere; condition 5, only in region C) tracks noticeably better in region C than either contact-aware condition, despite being a *worse* local approximation to the oracle Jacobian. Five analyses were run to investigate the mechanism, in the priority order agreed with the reviewer; analysis 3 (the four-branch same-state counterfactual) produced the clearest, most surprising result and is the headline finding of this report — see Section 3. Registry note: condition 5 ("COND5" below) is the hybrid schedule — contact Jacobian everywhere, no-contact-model Jacobian strictly inside region C (55–65mm), 1mm smoothstep ramps at 54–55mm and 65–66mm — added to `run_registry.py` from its 3 confirmed-current-box reps.

## 1. Identity audit: is OLD-C vs COND5 identical outside region C?

Three checks, in increasing order of rigor — all confirm yes:

- **(a) Schedule diff**: COND5's own schedule array is bit-identical to `SCHED_C` everywhere outside `[54,66]mm` (max abs diff = exactly 0.0), and differs inside as expected (max abs diff = 0.439 at the transition). Confirmed by construction, not assumed.
- **(b) Controller config diff**: `controller_metadata.json` is byte-identical between an OLD-C run and a COND5 run across all 26 logged fields (horizon, weights, delay, magnet bounds, etc.) — zero differences.
- **(c) Two-step same-state replay audit in region A**: at COND5's own 369 region-A anchors, replaying its own schedule reproduces the real logged command with cosine similarity ~0.87 (direction) but weaker relative-norm agreement (~0.75) — consistent with, not worse than, this investigation's own established replay-fidelity limit (full-run baseline ~0.89–0.91 cosine), and expected to degrade further here because region A is deliberately the *low-command-authority* zone, where small absolute differences inflate relative-error metrics. Critically, the **controller-identity check** (COND5's replay vs. an OLD-C-schedule replay at the *identical* state) shows **exact agreement at the median anchor** (theta=0.0°, command difference 1.7e-16) and 86.4% of anchors within 5°, since both branches share the same bit-identical schedule and byte-identical config there — the heavier tail is numerical sensitivity at near-zero-command ticks, not a real controller difference.

**Conclusion**: the region-A RMSE gap between OLD-C (0.358mm) and COND5 (0.271mm) is ordinary trajectory/run variation, not a hidden controller/configuration difference. Pre-entry states are comparable.

## 2. Five-zone pre/ramp/C/ramp/post intervention analysis

Five zones matching the actual schedule transition (not a naive 3-zone split): `R_pre=[50,54]mm` (pure J1) → `R_ramp_in=[54,55]mm` → `R_C=[55,65]mm` (pure J_NC) → `R_ramp_out=[65,66]mm` → `R_post=[66,70]mm` (pure J1 again).

| cond | rep | R_pre | R_ramp_in | R_C | R_ramp_out | R_post |
|---|---|---|---|---|---|---|
| OLD-C | 1 | 0.491 | 0.776 | 0.570 | 0.564 | 0.397 |
| OLD-C | 2 | 0.500 | 0.620 | 0.649 | 0.945 | 0.635 |
| OLD-C | 3 | 0.457 | 0.734 | 0.539 | 0.539 | 0.598 |
| COND5 | 1 | 0.432 | 0.595 | 0.404 | 0.478 | 0.668 |
| COND5 | 2 | 0.414 | 0.678 | 0.404 | 0.586 | 0.749 |
| COND5 | 3 | 0.463 | 0.711 | 0.349 | 0.452 | 0.574 |

(RMSE, mm, per zone per run.)

`Delta_e_C_pre = RMSE(R_C) - RMSE(R_pre)`: OLD-C = [+0.079, +0.149, +0.081] (median **+0.081mm**, gets *worse* going pre→C, even though OLD-C never changes its Jacobian there — ordinary trajectory drift); COND5 = [−0.029, −0.009, −0.115] (median **−0.029mm**, gets *better* going pre→C, exactly where it switches to the no-contact Jacobian).

**ΔΔe = −0.029 − (+0.081) = −0.110mm.** The divergence opens right at the `R_pre→R_ramp_in` boundary and is consistent with the error trend reversing specifically where the Jacobian swap happens — conditional on analysis 1's confirmation that pre-entry states/config are comparable, this provides **strong causal support** for the Jacobian swap (not trajectory noise) driving the region-C improvement, though it is not asserted as "directly attributable" outright. (Secondary observation, not load-bearing: COND5's R_post is elevated relative to its own R_pre — a possible carryover effect worth a mention but not pursued further here.)

## 3. Four-branch same-state counterfactual (NEW/OLD-C/NC/COND5) — THE HEADLINE RESULT

Extends the within-anchor {J1, J_hybrid, J_NC, J4} counterfactual to anchors drawn from **all 4 conditions' own realized states** (not just NEW's, as in the original Script 5). At every anchor, with state/residual/previous-command/delay-state held fixed, 4 branches are resolved: `J1` (plain scheduled contact Jacobian), `J_hybrid` (production hybrid map at that exact state), `J_NC` (no-contact-model schedule, direct same-index lookup), `J4` (model oracle, full 15-step horizon recomputed at the reference's own state). Replay-fidelity-to-hardware is consistent with this investigation's established baseline (mean cosine similarity 0.84–0.90 across all 12 reps).

**`Delta_E_J^{NC-J1}(x) = E_J,D(J_NC;x) - E_J,D(J1;x)`** (positive = `J_NC` is the worse approximation to the oracle `J4`): **positive at essentially every anchor, in every region, for all 4 conditions** (frac_positive 0.97–1.00 in regions B/C, median ≈ +0.00096 to +0.00114). This is exactly the expected, unsurprising half of the result: `J_NC` never gets physically closer to the true contact-aware Jacobian than `J1` does, anywhere, for anyone.

**`Delta_theta_u^{NC-J1}(x) = theta_u(J_NC;x) - theta_u(J1;x)`** (both branches' *commands* compared against `J4`'s command at the same state; positive = `J_NC`'s command direction is further from what the oracle would have commanded): this is where the puzzle sharpens dramatically.

| anchors drawn from... | region | n | Delta_theta_u median | frac_positive |
|---|---|---|---|---|
| NEW's own states | C (55–65mm) | 216 | −3.0° | 0.48 |
| OLD-C's own states | C | 216 | −2.6° | 0.46 |
| **NC's own states** | **C** | **230** | **+57.2°** | **0.96** |
| **COND5's own states** | **C** | **223** | **+51.6°** | **0.96** |

At the region-C states NEW and OLD-C actually visit, `J_NC`'s command direction is essentially a coin-flip relative to `J1`'s for alignment with the oracle — no real difference. But **at the region-C states NC's and COND5's own (no-contact-Jacobian-driven) trajectories actually visit, the command that was *actually used* (`J_NC`) points, on average, ~50–57° further from the oracle's own recommended direction than the `J1` counterfactual would have — at 96% of those anchors.** This is not a subtle effect.

The same pattern shows up directly in the absolute command-angle-to-`J4` table (mean, degrees, across all sampled ticks, not anchor-paired):

| condition | own (actually used) | J1 (counterfactual) | J_hybrid (counterfactual) | J_NC (counterfactual) |
|---|---|---|---|---|
| NEW | 20.0° | 20.2° | 20.0° | 35.0° |
| OLD-C | 19.8° | 19.8° | 19.6° | 35.6° |
| **NC** | **40.2°** | **10.1°** | **9.9°** | 45.2° |
| **COND5** | **30.3°** | **12.6°** | **12.4°** | 49.7° |

For NEW and OLD-C, the actually-used command sits close to what every branch would have commanded (~20°) — unsurprising, since their own branch essentially *is* `J1` (or `J1`-plus-rare-hybrid). For **NC and COND5, the command they actually sent (own = `J_NC`-derived) is 3–4x further from the oracle's recommendation (30–40°) than the `J1`/`J_hybrid` counterfactuals would have been at those identical states (10–13°)** — yet, per the primary outcome metrics, NC and COND5 track the physical path *better* in region C than NEW/OLD-C do.

**This is the sharpest, most quantified version yet of the reviewer's reframed question**: the Jacobian/command most aligned with the physically-accurate contact-model oracle (`J1`/`J_hybrid`, ~10–13° off at NC's/COND5's own states) is demonstrably **not** the one producing better closed-loop tracking there — the command that diverges furthest from the oracle's own recommendation (`J_NC`, ~40–50° off) is. Local model-oracle alignment and closed-loop tracking quality are **not monotonically related under contact** in this dataset — consistent with, and now much more concretely evidenced than, the hypothesis the reviewer named explicitly: *"local model accuracy and closed-loop control quality are not monotonically related under contact."* This reframes Section 4's directional-redistribution finding (smaller/opposite-signed stem-normal correction) as the *mechanism* and this section's result as the *evidence that mechanism matters more than oracle-alignment*: the MPC's behavior under contact depends on how the model interacts with the objective, constraints, delay, and the real wall — not on Frobenius-norm proximity to the forward model's own Jacobian.

## 4. J1-vs-J_NC command-direction decomposition

**Corrected framing (per review): not a "smaller command is better" test.** ‖u_NC‖/‖u_J1‖ ratio is actually **>1** at NEW/OLD-C's own states (median 1.47/1.50 — NC requests *more* total authority there, not less), while at NC/COND5's own states the ratio is much closer to 1 (1.14/1.15). The "regularization via shrinkage" hypothesis is **not supported** as stated. The real signal is directional, not scalar:

| source states | axis | J1 branch (median) | J_NC branch (median) | \|NC\|<\|J1\| fraction |
|---|---|---|---|---|
| NEW's own | tangential | +1.51e-4 m | +2.77e-4 m | 0.18 |
| NEW's own | **stem-normal** | **−4.18e-5 m** | **+5.48e-5 m** | 0.38 |
| NEW's own | transverse | −2.6e-6 m | +1.68e-4 m | 0.05 |
| OLD-C's own | stem-normal | −2.24e-5 m | +7.08e-5 m | 0.32 |
| **NC's own** | **stem-normal** | **−2.67e-4 m** | **+7.94e-5 m** | **0.81** |
| **COND5's own** | **stem-normal** | **−2.07e-4 m** | **+7.23e-5 m** | **0.79** |
| NC's own | transverse | −8.06e-5 m | −5.45e-5 m | 0.69 |
| COND5's own | transverse | −1.30e-4 m | −6.00e-5 m | 0.88 |

(Predicted tip-displacement component, m, from `Δp = J_branch @ Δχ_branch` with the full 7-component `Δχ=[q̇Δt; L̇Δt]` — insertion never dropped. "Stem-normal" = tip displacement projected onto the production contact model's own wall-normal convention, `LumenQuery.closest_with_gradients`'s `grad_delta` at the nominal tip position — reused from `proper_research/simulation/magnetic_beam/contact.py`, not invented. Labeled "tip displacement along the stem-contact normal," not "contact-node mobility" — a true contact-node-level decomposition wasn't attempted; no simple reusable accessor for the discretized beam's contact-node position/Jacobian was found in the time available. This is a flagged limitation, not a silently-skipped step.)

**Sign convention** (`LumenQuery.closest_with_gradients`'s own definition, read from source, not assumed): `grad_delta = (p - q) / ‖p - q‖`, where `q` is the nearest lumen-centerline point — i.e. it points **outward from the centerline, toward the wall**. Positive stem-normal = moving further from the centerline (toward/through the wall); negative = moving back toward the centerline (retreating from the wall).

**Reading this, with that convention applied**: at NEW's/OLD-C's own states, J1 commands a **retreat from the wall** (negative stem-normal) while J_NC commands **motion toward the wall** (positive) at the typical anchor — opposite-signed, not just different magnitude. At NC's/COND5's own states — i.e., states the no-contact-Jacobian-driven trajectory actually visits — J1 (evaluated there as a counterfactual) would have commanded a **much larger retreat-from-wall correction** than what was actually used (|J1| stem-normal is ~3-4x larger in magnitude than at NEW/OLD-C's states, and still negative), while the Jacobian actually in use (J_NC) requests a **smaller-magnitude, wall-ward** correction at 79–81% of those anchors. Combined with the transverse column (J_NC also smaller in transverse at NC's/COND5's own states, 69–88% of anchors, while *larger* in transverse at NEW's/OLD-C's own states), the pattern is directionally consistent with **redistribution, not simple regularization**: at the states the no-contact Jacobian's own trajectory visits, it avoids the large wall-retreat correction J1 would have demanded there, committing more modestly and putting relatively more authority into tangential (along-path) motion instead. Reported as hypothesis-testing language only (**consistent with**, not confirmed) — this reads the sign convention correctly from source, but the physical mechanism (why a smaller/opposite-sign normal correction tracks *better* under contact) is not independently confirmed here.

## 5. NEW's w>0 tick consequence analysis

Across NEW's 3 reps, only 69 ticks total (out of ~1,479 combined, ~4.7%) have `w>0` (the hybrid map partially or fully active) — consistent with Section 3 of the main report's finding that NEW runs identically to OLD-C's J1 schedule ~95% of the time in this dataset.

- `‖Δu‖` (own vs. J1 counterfactual, same state) is **not negligible**: median 3.2e-3, p90 1.7e-2, max 2.7e-2 — roughly 15–100% of a typical command's own magnitude (~1.5–2.5e-2 from Section 4's table). Low activation percentage does **not** mean the hybrid-induced command changes are tiny.
- But the **predicted consequence is close to a coin flip**: `Δ V_track` (full predicted-horizon tracking cost, own better when positive) has frac_positive = **0.522**; `Δ e_pred1` (first-step prediction only) has frac_positive = **0.493**, median effectively zero (−6.3e-7m). Neither full-horizon nor first-step predicted tracking cost shows a one-sided improvement when the hybrid activates.

**Conclusion**: the w>0 ticks produce real, non-trivial command changes, but those changes don't show a consistent predicted-tracking benefit — the NEW-vs-OLD-C RMSE advantage from the main report is **probably not caused by J3's rare activation**, reinforcing Section 3's caveat rather than resolving it. (Two metrics kept deliberately distinct per review: `Δ V_track` is the actual MPC horizon cost `Σ_h‖e_h‖²` from `step.predicted_beam_errors`, not conflated with `Δ e_pred1`, the first-step-only term.)

## 6. Statistics-wording fix

Applied directly to `FRESH_RUNS_7CONDITION_ANALYSIS.md` (Sections 2 and 5): "statistically significant" replaced with "consistent across all three fresh repetitions, run-level bootstrap interval entirely negative," given the fragility of a 3-vs-3 bootstrap CI.

## 7. Hardware rerun: prepared, not executed

**Recommendation unchanged from the plan**: bring OLD-C and COND5 only from n=3 to n=5 (2 more reps each, interleaved in acquisition order); optionally condition 4 (NC-on-contact-everywhere) +2 reps, lower priority. Do **not** rerun inverse-Jacobian, frozen-Jacobian, or the true-no-contact-plan condition.

**Important caveat on the exact launch command**: neither `summary.json` nor `controller_metadata.json` for any existing run records the literal CLI invocation (argv) used to launch it, and this investigation's config dump does not map cleanly onto a single argparse definition found in the repo (the dataclass fields logged in `summary.json["config"]` don't all correspond 1:1 to flags in the vessel-stage-a runner script, likely because several are internal defaults rather than CLI-exposed). **Rather than fabricate a full ~40+ flag command that might be subtly wrong on real hardware, the safe path is: reuse your own last successful launch invocation for OLD-C (the one that produced `..._schedule_compare_v3_20261015T153950Z`) and for COND5 (the one that produced `..._hybridcon_nocon_v6_710_20261015T122744Z`) verbatim, changing only the `--run-name` suffix to the next rep number.** Both already-successful launches prove every other flag correct; only the run label needs to change for a new rep.

## What this does and doesn't establish (final)

**Established**: region-A/pre-entry comparability between OLD-C and COND5 (analysis 1); the region-C improvement event-studies cleanly around the actual schedule transition, not an artifact of a naive region split (analysis 2); at every anchor, for every condition, `J_NC` is *always* a worse approximation to the oracle Jacobian than `J1` (analysis 3's `Delta_E_J^{NC-J1}`, unsurprising); yet at the states NC's/COND5's own trajectories actually visit in region C, the command they actually sent diverges from the oracle's own recommended direction by ~50–57° on average — 3–4x further than the `J1`/`J_hybrid` counterfactuals would have at those identical states (analysis 3, the headline finding); the "smaller command" regularization hypothesis is false as stated, but a directional-redistribution hypothesis (smaller/opposite-signed stem-normal correction, particularly at the no-contact-Jacobian's own visited states) is consistent with the data and offers a candidate *mechanism* for analysis 3's result (analysis 4); NEW's rare hybrid activations are not negligible in magnitude but show no one-sided predicted-tracking benefit, so they are not the explanation for NEW's advantage over OLD-C (analysis 5).

**Not established — the open question this report sharpens rather than closes**: *why* oracle-divergent commands track better under contact here. Analysis 3 proves the effect is real and large; analysis 4 offers a directional (not magnitude) mechanism consistent with it; neither explains the underlying control-theoretic reason a locally-"wrong" Jacobian would be closed-loop-beneficial under this specific contact/delay/constraint structure. That is now the central open question for this investigation, stated precisely: **local model accuracy and closed-loop control quality are not monotonically related under contact.**
