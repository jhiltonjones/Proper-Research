# Three-condition hardware campaign, n=5 per condition (2026-10-14/15)

NEW (contact MPC + smooth state-conditioned J1/J3 hybrid) vs OLD-C
(contact MPC + original scheduled contact Jacobian J1) vs NC (MPC +
scheduled no-contact Jacobian), 5 hardware repetitions each, floor240
vessel plan. Supersedes an earlier n=3 attempt, which was invalidated
by a timing regression in NEW (fixed before this campaign — see
`speed_fix_validation/` and the `contact_jacobian_hybrid.py`/
`delay_aware_mpc.py` commit this directory ships alongside).

Scripts in `scripts/` reproduce every number below from the raw
`close_loop_logs/myrun/` run directories named in `run_registry.py`.
Full per-script output is in `results/*.json`; `figures/` has the 7
main plots plus NEW's per-rep hybrid-operation breakdowns.

## Gate (must pass before anything below is causal, not just descriptive)

**PASS.** Timing/solver-status practical-equivalence, predefined
before inspecting these runs: median tick_dt diff = 0.50ms (tolerance
10ms), solver_time diff = 0.55ms (tolerance 10ms), realization-ratio
diff = 0.019 (tolerance 0.05), time-limit-QP-fraction diff = -0.2pp
(NEW actually lower than OLD-C). Zero `deadline_miss` and zero
hard-constraint violations across all 15 runs. One caveat: the new
TCP-workspace constraint's own configuration isn't recorded in
`controller_metadata.json` (a logging gap, not fixed retroactively),
so "configured identically across conditions" rests on knowing the
commands used, not a direct artifact check — see `results/
timing_gate_result.json` for the exact wording.

## Primary physical result (region C = [55,65]mm, fixed from the
## prior investigation, not tuned to these runs)

`Delta_C = median_r(RMSE_NEW,C) - median_r(RMSE_OLD-C,C) = -0.060mm`
(NEW better) — **PASS** on the primary criterion, but the run-level
bootstrap 95% CI (resampling the 5 runs, never ticks) is
`[-0.139, +0.015]mm`, which includes zero: not statistically
significant at n=5, though consistently in NEW's favor. No full
range separation at region-C level (NEW's worst rep ties OLD-C's
best); whole-run RMSE (not region-C-specific) *does* fully separate.

## Mechanistic result — genuine tension between two valid tests

- **Condition-level/aggregate** (mean curves across each condition's
  own states): everything lines up — `E_J,D` down, `theta_u` down
  (-1.12deg), `E_future` down, the full chronology
  (E_J,D -> theta_u -> E_future, all down) holds, and
  `r_q^NEW < r_q^OLD-C` in region C (configuration stays closer to
  plan).
- **Within-anchor, state-paired test** (same exact measured state,
  only the Jacobian source changes — the test this plan's review
  explicitly asked be treated as primary): at 95% of NEW's region-C
  anchors, NEW's own Jacobian is actually slightly *farther* from the
  model-oracle `J4` than plain J1 would have been at that same state
  (median `Delta_E_J = -0.00012`). Command angle is more mixed (62%
  of anchors favor NEW, median `+0.38deg`, but with a wide spread).

**Conclusion**: the condition-level comparison is confounded by which
states each condition happened to visit; once state is held fixed,
the clean "better local Jacobian -> better command" story mostly
disappears. Per the plan's own four-outcome classification: *"within-
anchor E_J,D/theta_u NOT consistently down -> the hybrid did not
deliver its intended model-level advantage on these anchors."* The
physical tracking improvement is real in direction but not yet
significant at n=5, and is not confidently attributable to Jacobian
model fidelity specifically.

## Specificity, safety, other pairwise comparisons

- NEW also improved in region A (not just C) — doesn't falsify
  specificity but means the benefit isn't cleanly localized.
- Safety: PASS, no violations; tightest margin ~1.65-1.85mm
  (`z_tight_max`, same face as before, unchanged by this round).
- OLD-C vs NC (contact-modelling value): OLD-C worse than NC in its
  own contact region (consistent with contact disturbance hurting the
  plain schedule).
- NEW vs NC (end-to-end gain): NEW still short of NC's region-C RMSE
  by +0.090mm median — contact-aware modelling has not yet closed the
  gap to simply not modelling contact at all, in this region.

## What would change this conclusion

A larger n (more hardware reps) to tighten the bootstrap CI and the
within-anchor fraction-positive estimate; or a direct investigation of
*why* the within-anchor and condition-level tests diverge so sharply
(e.g. whether a handful of outlier anchors dominate the aggregate
curves) — flagged as a natural next step, not yet done here.
