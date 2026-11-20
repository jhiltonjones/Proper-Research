# Fresh 7-Condition Hardware Comparison at Current Workspace Setting (2026-10-15/16)

This covers the specific runs the user supplied and asked to be analysed: three repetitions each of seven controller/Jacobian conditions, all executed **after** the final workspace-box edit of the day (`_WORKSPACE_XYZ_MIN_M=(0.27,-0.71,0.17)`, `_WORKSPACE_XYZ_MAX_M=(0.66,-0.0,0.5)`). Box membership for every run below was confirmed directly from each run's own `summary.json` → `config.workspace_xyz_min_m/max_m`, not inferred from timestamps.

## 1. The seven conditions and their runs

| # | condition | runs (timestamp suffix) |
|---|---|---|
| 1 | Inverse Jacobian controller (contact schedule, clip-not-abort, margins 15mm/15mm) | `v1_142604Z`, `v1_142710Z`, `v3_151224Z` |
| 2 | MPC, NEW schedule (hybrid_j1j3), contact | `v1_152223Z`, `v2_152601Z`, `v3_152906Z` |
| 3 | MPC, OLD schedule (plain scheduled contact Jacobian), contact | `v1_153149Z`, `v2_153423Z`, `v3_153950Z` |
| 4 | No-contact-model Jacobian evaluated on the contact plan | `v9_121417Z`, `v10_121753Z`, `v11_122200Z` |
| 5 | Hybrid: contact Jacobian everywhere, no-contact Jacobian only in region C | `v4_710_121045Z`, `v5_710_122458Z`, `v6_710_122744Z` |
| 6 | True no-contact (no-contact plan + no-contact schedule) | `v1_151447Z`, `v2_151717Z`, `v3_151941Z` |
| 7 | Frozen Jacobian (contact, LTI — linearized once at the start, never updated) | `v1_130501Z`, `v2_130706Z`, `v3_130859Z` |

Conditions 2/3/4 map directly onto the model-fidelity pipeline's NEW/OLD-C/NC-model labels (Section 3) — all three now share the identical 527-sample contact-plan reference, so no cross-plan index mapping is needed (unlike the earlier n=5 campaign, where "NC" ran on a separate 415-sample plan).

## 2. Primary tracking metrics

| condition | whole-run RMSE (median of 3) | region-C RMSE | region-C median err | region-A RMSE |
|---|---|---|---|---|
| 1. Inverse Jacobian | *not comparable — see note* | n=0 (never reached) | — | 0.43 (1/3 reps only) |
| 2. MPC NEW (hybrid_j1j3) | 0.791 | 0.488 | 0.337 | 0.388 |
| 3. MPC OLD (schedule) | 0.888 | 0.570 | 0.308 | 0.358 |
| 4. No-contact-model on contact plan | 0.994 | **0.393** | 0.298 | 0.253 |
| 5. Hybrid, no-contact only in region C | 0.973 | **0.404** | 0.284 | 0.271 |
| 6. True no-contact | 1.662 | 0.368 | 0.290 | 0.307 |
| 7. Frozen Jacobian (LTI) | *not comparable — see note* | n=0 (never reached) | — | 1.44 |

**Note on conditions 1 and 7**: two of the three inverse-Jacobian runs are extremely short (28 and 44 ticks — essentially just the opening transient before a stop), and even the longest run (347 ticks) never reached region C. The frozen-Jacobian condition fails consistently around tick 245–253 (of 527), also before reaching region C — confirmed reproducible across all 3 reps (see the earlier frozen-Jacobian analysis). Neither condition's whole-run/region-C numbers are directly comparable to the other five.

**Headline finding, replicated under this fresh, independently-confirmed workspace setting**: conditions 4 and 5 (no-contact Jacobian — either everywhere on the contact plan, or only in region C) both show markedly lower region-C RMSE (~0.39–0.40mm) than either real contact-aware condition (NEW 0.49mm, OLD-C 0.57mm). Condition 5's region-C RMSE (0.404) nearly matches condition 4's (0.393) despite only swapping the Jacobian in that one region — confirming the effect is genuinely localized to region C, not an artifact of a different trajectory.

**Condition 2 vs 3, region C** (same methodology as the earlier n=5 campaign): `Delta_C = median_r(RMSE_NEW) - median_r(RMSE_OLD-C) = -0.0819mm`, run-level bootstrap 95% CI `[-0.2029, -0.0329]mm` — entirely negative, excludes zero. Full range separation also holds (NEW's worst rep beats OLD-C's best). **Wording note (2026-10-16, per reviewer feedback): with only n=3 reps per condition, a bootstrap CI from 3 vs 3 runs is fragile, and "statistically significant" overstates what this supports. The safer description: the effect was consistent across all three fresh repetitions, and the run-level bootstrap interval was entirely negative.** This is nonetheless a tighter, cleaner result than the earlier n=5 campaign, where the equivalent CI included zero — see Section 3 below, though, for why this specific NEW-vs-OLD-C comparison should not be over-interpreted mechanistically.

## 3. A major caveat on the NEW-vs-OLD-C result: the hybrid map is barely active

Re-running the hybrid-operation-breakdown check (memorylessness-verified, same methodology as before) on these fresh NEW runs shows:

| rep | blend | pure_map | schedule_fallback (= identical to OLD-C's J1) |
|---|---|---|---|
| 1 | 3.0% | 1.0% | **95.9%** |
| 2 | 3.8% | 1.2% | **94.9%** |
| 3 | 3.9% | 1.0% | **95.1%** |

This is a large change from the earlier n=5 campaign (blend ~44–55%, fallback ~44–55%). Today's NEW condition is running the **same J1 schedule as OLD-C for ~95% of the whole path**, not just the early section. The consistent tracking advantage above is therefore very unlikely to be caused by the hybrid Jacobian map itself, since it's barely active. Most plausible explanations: run-to-run/hardware variability, or some other difference between the two runs unrelated to the contact-Jacobian model.

## 4. Model fidelity (normalized E_J,D) on this fresh dataset

Using the now-normalized metric (`E_J,D = ‖(J_used−J4)·D‖_F / ‖J4·D‖_F`, addressing the earlier reviewer critique that this must be normalized, not raw) on conditions 2/3/4, condition-level, J4 recomputed at the real measured state every sampled tick. **"NC" here means condition 4 specifically — the no-contact Jacobian MODEL evaluated on the CONTACT plan's own reference (same 527-sample indexing as NEW/OLD-C) — NOT condition 6 (the true no-contact plan + no-contact schedule).** The underlying `run_registry.py` "NC" entries for this dataset point at condition 4's three runs (`v9`/`v10`/`v11`), and `plan_root_for()` was set to the contact plan unconditionally, so J4 is computed on the identical contact-plan geometry for all three rows below.

**Whole sampled path:**

| condition | mean | median |
|---|---|---|
| NEW (condition 2) | 0.674 | 0.487 |
| OLD-C (condition 3) | 0.806 | 0.496 |
| NC = no-contact-model on contact plan (condition 4) | 0.980 | 0.810 |

**Region C only (s=55–65mm):**

| condition | mean | median |
|---|---|---|
| NEW (condition 2) | 0.633 | 0.471 |
| OLD-C (condition 3) | 0.877 | 0.482 |
| NC = no-contact-model on contact plan (condition 4) | 0.888 | 0.854 |

**Reading this alongside Section 3**: NEW shows consistently lower (better) E_J,D than OLD-C, both overall and in region C — but this is a condition-level comparison (each condition evaluated on its own realized states, not state-paired), and given NEW uses an identical Jacobian to OLD-C on ~95% of ticks, the gap is puzzling rather than a clean confirmation that the hybrid map improves fidelity. Two candidate explanations, neither yet checked: (a) the ~5% of ticks where NEW's Jacobian genuinely differs from OLD-C's carry outsized weight in the aggregate mean/median; (b) real tracking-state differences between the two runs (not Jacobian choice) are driving part of the gap, the same "condition-level is confounded by differing states" issue flagged throughout this investigation. This has **not** been resolved — it's an open question, not a conclusion.

NC (no-contact-model) is worst by a wide margin in both views, as expected since it omits contact physics entirely. The large mean-vs-median gap for OLD-C in region C (0.877 vs 0.482) indicates a handful of high-error ticks are pulling the mean up substantially; the median is the more representative summary here.

## 5. What this does and doesn't establish

**Established**: the physical tracking advantage of the no-contact Jacobian in region C (conditions 4 and 5) replicates cleanly under this fresh, independently-verified workspace setting. The condition 2 vs 3 comparison is tighter than the earlier n=5 campaign's equivalent (bootstrap CI entirely negative here, vs. including zero before), but per the wording note in Section 2, this is described as a consistent effect across all three repetitions rather than as statistical significance, given n=3 per condition.

**Not established**: why NEW's hybrid-map condition outperforms OLD-C's plain schedule — the aggregate E_J,D numbers favor NEW, but the hybrid map is active on <5% of ticks in this dataset, so "the hybrid Jacobian is more accurate" cannot be the explanation for most of the observed gap. This mirrors the earlier n=5 campaign's unresolved tension (condition-level mechanistic evidence vs. the more rigorous within-anchor paired test), now reproduced with an even starker version of the same puzzle.

**Not yet done on this fresh dataset**: the within-anchor, same-state paired counterfactual (the rigorous test that, on the earlier n=5 dataset, found the hybrid Jacobian was *not* closer to the oracle at 95% of NEW's own region-C anchors). Repeating that specific test here would directly address the puzzle in Section 4.
