# Rectangle supplementary material

All numeric outputs here were regenerated directly from the saved offline
replay/ablation scripts (not retyped) and cross-checked against the numbers
reported during the investigation -- every one matched exactly.

## (a) Original INV-6 (joint-only inverse) comparison — the confounded baseline
- `mechanism_figure.png`, `mechanism_figure_SJ.png`, `mechanism_figure_FJ.png`
- `factorial_main_figure.png`, `interaction_plot.png`
- Table: `../tables/run_level_summary_INV6.md`

This is the ORIGINAL comparison (before the actuator-authority confound was
identified and fixed with INV-7). Kept for the record and because it's what
motivated the whole matched-authority investigation — not the number to cite
as the final result.

## (b) Unit validation of the INV-7 matched-authority controller
- `unit_tests_inv7.txt` — all 6 required pre-hardware checks (exact
  reduction to INV-6 when insertion is disabled, x/y authority split,
  J*du direction, mm-vs-m scale invariance, hard limits respected)
- `raw_vs_normalized_pinv_diagnostic.txt` — demonstrates why raw
  (un-normalized) DLS would have been unit-dependent

## (c) Actuator-authority confound: how much of INV-6-vs-MPC was authority?
- `../tables/run_level_summary_INV6.md` vs `run_level_summary_INV7_matched.md`
- `mechanism_figure_INV7_SJ.png`, `mechanism_figure_INV7_FJ.png` (in ../figures/)
- `authority_ladder.png`, `interaction_plot_extended.png` (in ../figures/) —
  INV-6 -> INV-7 -> MPC progression

## (d) Jacobian staleness mechanism (why FJ hurts the inverse law specifically)
- `jacobian_geometry_drift.txt`, `jacobian_geometry.png` — directional
  rotation of the normalized joint-to-y Jacobian row vs the frozen (index-0)
  snapshot; correlates with the realized FJ lateral-error gap

## (e) Constraint-clipping mechanism (ruled out as dominant)
- `clip_angle_diagnostic.txt`, `inv7_clip_diagnostic.png`

## (f) Disturbance persistence (beta_d) ablation — offline, frame-corrected
- `beta_disturbance_persistence_ablation.txt`
- Caveat stated in the file and in MANIFEST.md: offline counterfactual
  replay evidence, not a measured live effect

## (g) Insertion-feedback-specific ablation (MPC-full vs MPC with u_L pinned)
- `mpc_insertion_feedback_ablation_h3.txt` (horizon +3)
- `mpc_insertion_feedback_ablation_h10.txt` (horizon +10, robustness check)

## (h) Horizon length (N) sweep
- `horizon_sweep_analysis.txt`, `horizon_u0_agreement.png`
- N=3/5/10/15, beta_d=1.0 locked throughout

## (i) kp fairness check for INV-7
- `inv7_kp_fairness_sweep.txt` — kp in {0.6, 0.8, 1.0, 1.2}, confirms kp=1
  is not a poor choice (flat, shallow sensitivity)

## (j) INV-7 vs INV-6 offline replay (before the live pilot was run)
- `inv7_vs_inv6_offline_replay_SJ.txt`, `inv7_vs_inv6_offline_replay_FJ.txt`
