# Four Pairwise Comparisons for Write-Up (n=5 data, 2026-10-19)

All numbers below are drawn directly from the n=5 primary-metrics run already reported in `SIX_CONDITION_N5_ANALYSIS.md` — nothing new was computed. RMSE and lag are in mm; "lag" is reported as the median of each run's own mean-signed lag, taken in magnitude (|lag|) since sign here just indicates ahead/behind the time-parameterized reference, not an error magnitude. Region boundaries: A = [12,17]mm (early, low-authority), C = [55,65]mm (the established contact-sensitive zone), B = everything else, whole-run = no clipping.

**Headline caveat up front, so it doesn't get lost**: three of your four claims hold cleanly across the board. The contact-vs-no-contact-Jacobian claim (#1) holds whole-run and in region B, but **reverses in region C** — this is not a data-quality issue, it's the single most-investigated, most-replicated finding in this whole campaign (see `FOLLOWUP_MECHANISM_ANALYSIS.md`), and a reviewer who has seen the region-C numbers elsewhere would notice if a paper claimed "contact Jacobian is more accurate" without qualification. I'd present #1 as "more accurate everywhere **except** the near-wall region, where the opposite holds — and we have a mechanistic explanation for why," rather than as an unqualified accuracy win. The details are below.

## 1. Contact Jacobian (OLD-C) vs. No-Contact Jacobian (NC) — same MPC, same trajectory, Jacobian source only

| | whole-run RMSE | region A RMSE | region B RMSE | **region C RMSE** | whole-run \|lag\| | region C \|lag\| |
|---|---|---|---|---|---|---|
| Contact (OLD-C) | **0.963** | 0.272 | **1.045** | 0.643 | **0.174** | **0.545** |
| No-contact (NC) | 0.996 | 0.269 | 1.112 | **0.393** | 0.192 | 0.746 |

**Holds as claimed**: whole-run accuracy (contact 0.96 vs no-contact 1.00mm) and lag everywhere tested (contact has less lag both whole-run and even within region C, where it's otherwise losing on RMSE). Region A is essentially tied.

**Does not hold in region C**: the no-contact Jacobian is *more* accurate there (0.393 vs 0.643mm, NC's best region across the whole comparison), not less. This is a robust, now twice-replicated (n=3 and n=5) result with its own dedicated mechanistic investigation: at the states the no-contact-Jacobian trajectory visits in region C, its commands diverge ~50–57° from what the contact-model oracle itself would recommend at the identical state — i.e. it is a *worse* local model, yet produces *better* tracking there. Full derivation in `FOLLOWUP_MECHANISM_ANALYSIS.md`, Section 3.

**Suggested framing for the paper**: "The contact-aware Jacobian gives higher whole-path accuracy and lower lag than the no-contact Jacobian, consistent with modeling the true physical coupling. The one documented exception is the near-wall region, where [mechanism finding] — a result we investigate separately in Section X."

## 2. Scheduled Jacobian (OLD-C) vs. Frozen Jacobian (FROZEN) — same MPC, same trajectory, update policy only

| | whole-run RMSE | region A RMSE | reaches region C? |
|---|---|---|---|
| Scheduled (OLD-C) | **0.963** | **0.272** | yes, 5/5 |
| Frozen (FROZEN) | 2.371 | 1.440 | **no, 0/5** |

**Holds cleanly, no exceptions.** The frozen (LTI) Jacobian is categorically worse — roughly 2.5x the whole-run RMSE and 5.3x the region-A RMSE — and never reaches region C at all across any of its 5 reps, always failing on accumulated tracking error at s≈31–41mm. Notably the degradation is visible even in region A (just 12–17mm into a ~75mm path), showing the Jacobian goes stale almost immediately once it stops being updated, not only once contact effects appear later on.

## 3. Offline Contact-Modeled Plan (OLD-C: contact plan + contact schedule) vs. Offline No-Contact-Modeled Plan (TRUE_NC: no-contact plan + no-contact schedule)

| | whole-run RMSE | region A RMSE | region C RMSE |
|---|---|---|---|
| Contact plan (OLD-C) | **0.963** | 0.272 | 0.643 |
| No-contact plan (TRUE_NC) | 2.088 | 0.283 | **0.505** |

**Holds whole-run**: the contact-modeled offline plan is more than twice as accurate overall (0.96 vs 2.09mm) — the dominant effect here. Region A is essentially tied.

**Reverses in region C, same direction as comparison #1** (TRUE_NC 0.505 vs OLD-C 0.643mm) — but note TRUE_NC is still clearly *worse* than NC's region-C result (0.393mm) despite both using a no-contact Jacobian. That gap (TRUE_NC vs NC) is itself informative: it shows the region-C benefit isn't simply "ignore contact physics" — it specifically requires applying the no-contact Jacobian to the *contact plan's own* trajectory, not to a differently-shaped no-contact-designed path. See `SIX_CONDITION_N5_ANALYSIS.md` Section 5 if you want that caveat in the writeup too.

One methodological note for the paper: TRUE_NC runs a geometrically different path (its own no-contact plan, ~384 ticks vs ~491 for the contact-plan conditions), so "whole-run" here compares each condition's own full traversal, not a tick-for-tick identical trajectory — the regions (defined by physical progress s, not tick index) are the fairer basis for the near-wall claim specifically.

## 4. MPC (OLD-C) vs. Inverse-Jacobian Controller (INVJAC)

| | whole-run RMSE (median) | whole-run RMSE (mean) | region A RMSE | reaches region C? |
|---|---|---|---|---|
| MPC (OLD-C) | **0.963** | **0.962** | **0.272** | yes, 5/5 |
| Inverse-Jacobian (INVJAC) | 2.007 | 3.037 | 0.419 (n=3 of 5) | **1/5 only** |

**Holds cleanly.** MPC is roughly 2–3x more accurate on whole-run RMSE (more on the mean than the median, because 2 of the 5 inverse-Jacobian reps fail almost instantly and inflate the mean — this instability is itself part of the comparison, not something to average away quietly). Region A, where 3/5 inverse-Jacobian reps actually get going, still shows MPC ~35% more accurate. Only 1 of 5 inverse-Jacobian reps reaches region C at all (vs 5/5 for MPC), so there is no meaningful region-C accuracy comparison to report for this pair — I would not put a region-C number in the paper for INVJAC; note its absence instead, which is itself the finding (MPC's hard in-QP constraints keep it running long enough to matter; the inverse-Jacobian controller's unconstrained nullspace drift means it usually doesn't survive that long).

## Numbers behind every row above, for your own sanity-check against the source JSON

`comparison_out/phase1_2_six_condition_n5_results.json`, keys `per_run.<COND>_rep<N>.<region>.{rmse_mm, mean_lag_mm}`, region names `whole_run`/`A`/`B`/`C_primary`.
