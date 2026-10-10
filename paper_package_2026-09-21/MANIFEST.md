# Magnetic beam-steering control study — paper package (2026-09-21, restructured 2026-09-22)

Matched-authority inverse-Jacobian (INV-7) vs receding-horizon MPC, live on
a UR10e + electromagnet + camera-tracked soft magnetic composite beam.
Every run uses the same frame-corrected harness, the same
`|u_q|<=0.1 rad/s`, `|u_L|<=2mm/s` actuator limits, the same delay-aware
(d=2) preview, and the same normalized-DLS matched-authority inverse
controller (`kp=1`, validated — see
`01_rectangle/supplementary/inv7_kp_fairness_sweep.txt`). n=5 live reps per
condition unless noted otherwise. **INV-6 (joint-only feedback, the
original/confounded baseline) appears ONLY in supplementary S1, to justify
why INV-7 is used as the baseline everywhere else — it is not a main-paper
competitor.**

Figure/panel text is deliberately minimal (short sentence-case labels
only) — the experimental design and interpretation live in
`00_main/captions.md`, ready to paste into the manuscript as Fig./Table
captions. This is a deliberate split: graphic = data + concise labels;
caption = experimental design + interpretation. Causal discussion (e.g.
"consistent with Jacobian rotation driving INV-7's lateral failure")
belongs in the Results text, not the caption — see this file's caveats
section and each shape's `tables/*.md` for that fuller analysis.

## Status

| shape | INV7-FJ | INV7-SJ | MPC-FJ | MPC-SJ | status |
|---|---|---|---|---|---|
| **Rectangle** (10x15mm) | n=5 | n=5 | n=5 | n=5 | complete 2x2, plus full mechanism analysis |
| **Triangle** (25mm base, bimaterial) | n=5 | n=5 | n=5 | n=5 | complete 2x2 |
| **U-shape** (10x15mm, bimaterial) | n=5 | n=5 | **NA** (2 safety-terminated) | n=5 | 3/4 cells; MPC-FJ is a genuine failure mode |

NOT in this package (insufficient n, do not resurrect without collecting to
n=5): 20mm-base triangle (superseded by 25mm-base), S-curve (n=1 pilot).

## Package structure

```
00_main/            <- MAIN PAPER figure/table set (Fig. 1-5, Table I-II; see below)
                        captions.md holds all Fig./Table caption text
01_rectangle/
  supplementary/    <- S1 + rectangle-specific mechanism/ablation detail (S2, S5-S10)
  tables/           <- full run-level numeric tables (INV6 + INV7-matched)
02_triangle_25mm/
  supplementary/    <- S3: full factorial x/y detail + mechanism deep-dive
  tables/
03_ushape/
  supplementary/    <- S4: full factorial x/y detail + mechanism deep-dive
  tables/           <- + MPC-FJ_excluded_notes.md, partial_contrasts.md
```

## Main paper figure/table set (00_main/)

A photo/schematic of the physical rig itself (robot + electromagnet + beam
+ camera) is still not included — that would need an actual photo or a
hand-built CAD/diagram asset, neither available in this session. Fig. 1
below covers the other half of orientation (what was commanded and how
closely it was followed spatially), built entirely from existing logs, no
new experiment. If a rig photo becomes available later it can be added
alongside Fig. 1 without renumbering anything else.

**Fig. 1** (`Figure1_spatial_path_tracking.png`) — spatial path tracking,
scheduled J only, no error quantification. One panel per shape (rectangle,
triangle, U-shape): desired tip path (black dashed), INV-7-SJ (green) and
MPC-SJ (blue) trajectories, individual runs faint + run mean thick,
start/end markers, equal-aspect task-plane mm coordinates. Answers "what
was commanded and how closely did the two fair controllers follow it,"
before any quantitative comparison. Deliberately SJ-only, no FJ traces, no
error overlays — orientation, not another analysis plot.

**Fig. 2** (`Figure2_cross_shape_SJ_performance.png`) — the headline
result: RMS tracking error, normalized feedback effort, and joint-rate
constraint activity, grouped by shape, INV-7 (hatched) vs MPC (solid),
individual points + mean±95% CI, shared legend. Full caption in
`captions.md`.

**Fig. 3** (`Figure3_FJ_SJ_robustness.png`) — model-mismatch robustness,
INV-7 vs MPC only (no INV-6), RMS only, one panel per shape, individual
points + mean±95% CI at every FJ/SJ endpoint (no numeric value labels —
the axis and Table II already carry the numbers). U-shape's MPC-FJ arm is
a concise "both attempts safety-terminated" annotation, not a fabricated
data point. Full caption in `captions.md`.

**Fig. 4** (`Figure4_rectangle_mechanism.png`) — rectangle mechanism deep
dive, one consolidated figure (A/B/C full-width, D split into two narrow
side-by-side sub-panels D1/D2 rather than one dual-y-axis panel with bars):
A) scheduled-J tracking, B) frozen-J tracking, C) frozen-J-only lateral
Jacobian rotation θ_Jy(k) (renamed from φ_y to avoid clashing with S10's
manipulability φ) + lateral-error gap overlay, D1) fraction of ticks with a
useful causal-+3 lateral correction (%), D2) net causal-+3 lateral-
correction efficiency (mm per unit normalized joint effort) — both D1/D2
show only individual run points + mean±95% CI (no bars, no dual axis),
grouped SJ | FJ with INV-7/MPC labels rather than four long compound
condition labels. The analogous scheduled-J rotation panel was
deliberately dropped — under SJ, INV-7 uses J(k) not J(0), so rotation
away from J(0) is not a model error there; see the caveats section below
for the full reasoning. Old Panel 2 (axis-resolved x/y mean error) moved to
`01_rectangle/supplementary/supp_axis_resolved_error.png`; the original
prediction-realization-scatter and MPC-only-actuator-trace panels remain
at S6/S7. Naming standardized throughout the main figure set: `INV-7`
(hyphen, no underscore) with the schedule omitted from the legend wherever
the panel/axis already establishes it (Figs. 1-3, Fig. 4 A-C); only Fig. 4
D1/D2 needs all four conditions, shown via SJ/FJ grouping rather than
compound labels like `INV7-SJ`.

**Fig. 5** (`Figure5_disturbance_horizon_ablations.png`) — two rectangle
controller-mechanism ablations, both offline counterfactual replay of the
same 5 recorded live trajectories (stated in `captions.md`, not in the
image): A) disturbance-persistence ablation, a paired dot/line plot (each
trajectory connects β_d=0→1, SJ and FJ grouped separately) rather than a
bar chart, since the design is paired, not independent-groups; B) horizon
saturation, with the N=3 opposite-sign tail (4.5% of ticks) annotated
directly since the median cosine agreement (0.97) alone would hide it.

**Table I** (`TableI_primary_SJ_results.md`) — shape × {INV-7 RMS, MPC RMS,
INV-7 $E_u$, MPC $E_u$}, mean ± SD, + reduction percentages. Source:
`*_run_level_summary.pkl`, condition keys `INV7-SJ` / `MPC-SJ` (provenance
kept here, not in the table itself — journal tables shouldn't carry
package-internal file paths).

**Table II** (`TableII_FJ_SJ_robustness_ratio.md`) — shape × {INV-7 FJ/SJ
ratio, MPC FJ/SJ ratio} computed from condition MEAN RMS (not a run-paired
ratio with propagated uncertainty — matters most for U-shape INV7-FJ,
whose mean is dominated by one outlier run, see caveats below). U-shape MPC
cell marked "NA — both attempted FJ trials safety-terminated" (not phrased
as an estimated probability).

## Terminology note (applies throughout)

**`Delta_law` is reserved for the factorial average** (MPC-INV7 averaged
over both FJ and SJ). **`Delta_law|SJ`** ("SJ simple effect") is the
matched-authority comparison under the scheduled Jacobian specifically —
the more important primary comparison, because SJ is the intended, valid
operating model; FJ is a model-mismatch robustness stress test, not a
second "equal" arm of the design. Quoting a factorial average as "MPC's law
advantage" overstates the effect under the actual scheduled controller
(worst on the rectangle: factorial −0.800mm vs the true SJ simple effect of
−0.240mm). Per-shape simple-effects tables are in each shape's `tables/*.md`.

## Supplementary map (S1-S10)

- **S1** `01_rectangle/supplementary/S1_authority_ladder.png` — INV-6 →
  INV-7 → MPC. The ONE place INV-6 appears; justifies using INV-7 as the
  baseline everywhere else. One-sentence version for the main text: "Giving
  the inverse controller insertion feedback reduced rectangle SJ RMS from
  1.213 to 0.784mm; all subsequent comparisons therefore use the
  matched-authority INV-7 baseline." Also: `interaction_plot.png` and
  `run_level_summary_INV6.md` (the pre-authority-fix 4-condition baseline).
- **S2-S4** full per-shape factorial x/y-RMS detail (superseded as the
  main headline by Fig. 2/3, kept for mechanistic detail):
  - S2: `01_rectangle/supplementary/factorial_main_figure.png` (now INV-7,
    not INV-6) + `interaction_plot_INV7_vs_MPC_xy_detail.png` (x/y-RMS
    breakdown, INV-7 vs MPC only).
  - S3: `02_triangle_25mm/supplementary/factorial_main_figure_tri25.png` +
    `interaction_plot_tri25.png`.
  - S4: `03_ushape/supplementary/factorial_main_figure_ushape.png`.
- **S5** `01_rectangle/supplementary/clip_angle_diagnostic.txt` +
  `inv7_clip_diagnostic.png` — constraint clipping is NOT the dominant
  mechanism for the inverse's FJ failure.
- **S6-S7** `01_rectangle/supplementary/mechanism_figure_INV7_{SJ,FJ}_original_panels3-4.png`
  — the original prediction-realization scatter (S6) and MPC-only actuator
  allocation traces (S7), i.e. Fig. 4's superseded Panels 3-4, kept for
  reference. Also triangle/U-shape's own full 4-panel mechanism figures
  (`02_triangle_25mm/supplementary/mechanism_figure_tri25_*.png`,
  `03_ushape/supplementary/mechanism_figure_ushape_*.png`) — these still use
  the ORIGINAL Panel 3/4 design (not the Fig. 4 replacement), since the
  θ_Jy(k)/directional-reliability rebuild was only done for the rectangle
  (where the mechanism was established).
- **S8** corner/straight RMS columns in each shape's `tables/*.md` — useful
  descriptively on the rectangle, does NOT generalize (U-shape INV7-FJ:
  corner=1.07mm, straight=2.06mm, despite the problem visibly starting at
  the corner — error from a bad transition persists into the following
  straight segment; triangle has cases where corner RMS is LOWER than
  straight RMS). Don't use as a cross-shape claim.
- **S9** `01_rectangle/supplementary/inv7_kp_fairness_sweep.txt` — INV-7's
  kp=1 gain validated against a small offline sweep.
- **S10** `01_rectangle/supplementary/manipulability_ablation/README.md` —
  manipulability/authority-shaping MPC secondary objective, investigated
  and CLOSED as a negative result (2026-09-22): the nonlinear authority
  landscape is real and exploitable, but its reference-linearized
  first-order QP surrogate mispredicts the sign of the true authority
  change in 40-43% of replayed ticks (a curvature/step-size failure, not
  gradient staleness — directly tested and rejected), and increasingly
  degrades tracking (+90% E_chosen at +10 steps) and effort (+163%) as its
  weight increases. The validated `Q_N` reference-nullspace regularizer was
  retained; the live controller was never modified. Suggested one-sentence
  main-text version: "We also investigated replacing reference-nullspace
  regularization with a projected log-volume authority reward. Although the
  nonlinear authority metric possessed exploitable nullspace gradients
  along the nominal trajectory, its reference-linearized first-order
  surrogate frequently mispredicted the sign of the true authority change
  in replay and increased tracking error and control effort as its weight
  increased. We therefore retained the reference-nullspace regularizer."

Also in `01_rectangle/supplementary/` but not separately numbered above:
`supp_axis_resolved_error.png` (Fig. 4's demoted old Panel 2, x/y mean
error vs path progress, SJ and FJ side by side),
`jacobian_geometry.png`/`jacobian_geometry_drift.txt` (Jacobian-schedule
deviation from frozen, the raw-data basis for Fig. 4's Panel C),
`beta_disturbance_persistence_ablation.txt`/`horizon_sweep_analysis.txt`/
`horizon_u0_agreement.png` (raw data basis for Fig. 5),
`directional_reliability_v2.{png,txt}` (raw data basis for Fig. 4 Panels D1/D2),
`mpc_insertion_feedback_ablation_h{3,10}.txt`,
`raw_vs_normalized_pinv_diagnostic.txt`, `unit_tests_inv7.txt`,
`inv7_vs_inv6_offline_replay_{FJ,SJ}.txt`.

## Known caveats to carry into the paper (Results-text material, not caption material)

- $E_u$ is dimensionless/normalized (S_u^-1-scaled deviation from
  $u_{\rm ref}$ — "normalized feedback effort," not "total actuator
  effort"), NOT millimetres.
- The disturbance-persistence (Fig. 5A) and horizon-saturation (Fig. 5B)
  findings are offline counterfactual-replay evidence of the same 5
  recorded live trajectories, not new live-trial measurements — state as
  "offline mechanistic evidence," not a measured effect size.
- Prediction-realization correlation (the ORIGINAL Panel 3, now S6/S7) does
  NOT consistently track controller performance across shapes: rectangle-SJ
  supports the MPC story (r_MPC=0.45 vs r_INV7=0.24) but triangle-SJ
  reverses it (r_MPC=0.26 vs r_INV7=0.43), as does triangle-FJ (0.13 vs
  0.34); on U-FJ, INV-7's correlation is higher than its own SJ value
  despite much worse tracking. Each controller is evaluated on states it
  created itself, so this isn't a pure policy comparison from a common
  initial state — the same-state/same-trajectory counterfactual replays
  (Fig. 5, insertion-limit sweep, manipulability A/B/C) are stronger causal
  evidence.
- Fig. 4 Panel C interpretation, for the Results text: the co-rise of
  θ_Jy(k) and the INV-7–MPC lateral-error gap from ~40% path progress,
  both peaking near 80–85%, is consistent with Jacobian rotation driving
  INV-7's lateral failure under a frozen model — this causal reading
  belongs in Results, not the figure caption.
- Corner-vs-straight RMS segmentation (S8) does not generalize past the
  rectangle — see S8 above.
- U-shape MPC-FJ: report as "both attempted trials developed rapidly
  increasing error in the same path region and were safety-terminated;
  therefore no complete-path RMS is reported" — do NOT imply a measured
  failure probability from n=2 attempts, and do not say "fails outright"
  (overstates what 2 attempts establish). The rectangle-triangle-U-shape
  co-localization of INV7-FJ/MPC-FJ trouble at the same path region is a
  **failure-location observation**, not an independent demonstration of the
  Jacobian-rotation root-cause mechanism (that mechanism — Fig. 4 Panel C —
  was established rigorously only on the rectangle, and only for the FJ
  condition specifically, not SJ).
- U-shape INV7-FJ has high dispersion (mean±sd 1.95±1.31mm from per-run
  values [0.84, 1.37, 1.48, 1.56, 4.52]mm) — one outlier run dominates the
  mean; median is 1.48mm. Keep individual points visible in any figure
  using this condition (Fig. 3 now does).
- Triangle: MPC-SJ vs INV7-SJ (−0.067mm) is a real but much smaller effect
  than the rectangle (−0.240mm); with n=5 the per-condition 95% CIs
  visibly overlap — don't oversell statistically. If the trial structure
  supports paired/block-aware inference, prefer that over treating the 10
  runs as independent groups. The schedule effect is asymmetric: INV-7
  benefits far more from scheduling (0.751→0.624mm) than MPC does
  (0.580→0.556mm) — MPC is comparatively schedule-insensitive here, same
  qualitative direction as the rectangle, much weaker magnitude.
- Cross-shape claim wording: "MPC produced lower mean scheduled-J RMS error
  on all three tested geometries, with a large effect on the rectangle and
  smaller effects on the triangle and U-shape" — not three independent
  proofs of the same effect size. A run-level hierarchical/block-aware
  model across the three geometries (if run ordering supports it) would
  strengthen this beyond the current per-shape n=5 CIs.
- Every quantitative table is regenerated directly from the saved
  `*_run_level_summary.pkl` files (single source of truth), not manually
  transcribed.
