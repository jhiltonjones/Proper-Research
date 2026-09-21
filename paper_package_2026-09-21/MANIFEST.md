# Magnetic beam-steering control study — paper package (2026-09-21)

Matched-authority inverse-Jacobian (INV-7) vs receding-horizon MPC, live on
a UR10e + electromagnet + camera-tracked soft magnetic composite beam.
Every run in this package uses the same frame-corrected harness, the same
`|u_q|<=0.1 rad/s`, `|u_L|<=2mm/s` actuator limits, the same delay-aware
(d=2) preview, and the same normalized-DLS matched-authority inverse
controller (`kp=1`, validated against a small offline sweep — see
`01_rectangle/supplementary/inv7_kp_fairness_sweep.txt`). n=5 live reps per
condition unless noted otherwise.

## Status: 3 shapes with complete or near-complete data

| shape | INV7-FJ | INV7-SJ | MPC-FJ | MPC-SJ | status |
|---|---|---|---|---|---|
| **Rectangle** (10x15mm) | n=5 | n=5 | n=5 | n=5 | complete 2x2, plus full INV-6 baseline + deep mechanism analysis |
| **Triangle** (25mm base, bimaterial) | n=5 | n=5 | n=5 | n=5 | complete 2x2 |
| **U-shape** (10x15mm, bimaterial) | n=5 | n=5 | **NA** (2 diverged) | n=5 | 3/4 cells; MPC-FJ is a genuine failure mode, not missing data |

Two more shapes were started but are NOT in this package (data not yet
sufficient for a paper claim):
- **Triangle 20mm** (original, non-widened base): only n=1-2 per condition,
  early numbers looked like a near-tie/INV-7-edge but that read did not
  survive to n=5 on the 25mm-base triangle (see `02_triangle_25mm/` — treat
  the 20mm triangle as superseded by the 25mm-base result, not as a fourth
  data point).
- **S-curve** (10x20mm, bimaterial): only n=1 SJ pilot. Not included.

## Folder structure

```
01_rectangle/
  figures/        <- MAIN figures: matched-authority INV-7 vs MPC (consistent
                     comparison with the other 2 shapes)
  tables/         <- run_level_summary_INV7_matched.{csv,md} is the one to
                     cite alongside triangle/ushape; run_level_summary_INV6.md
                     is the original (superseded) joint-only-inverse baseline
  supplementary/  <- (a) the original INV-6 4-condition figures/table (shows
                     the actuator-authority confound this whole investigation
                     started from), and (b) the deep mechanism analysis that
                     explains WHY MPC wins: actuator authority, Jacobian
                     staleness, disturbance persistence, horizon length,
                     clipping, kp fairness -- see supplementary/README.md
02_triangle_25mm/
  figures/, tables/   <- full 2x2, same structure as rectangle's main figures
03_ushape/
  figures/, tables/   <- 3 valid conditions + MPC-FJ_excluded_notes.md
                         explaining the NA cell
```

## Headline result across all 3 shapes

MPC beats the matched-authority inverse controller (Delta_law, MPC-INV7,
run-level means) on every shape at the scheduled-Jacobian (SJ) condition,
always using LESS normalized actuator effort (E_u):

| shape | Delta_law E_RMS (SJ) | E_u: INV7-SJ | E_u: MPC-SJ |
|---|---|---|---|
| Rectangle | -0.240mm (INV7=0.784, MPC=0.544) | 0.625 | 0.369 |
| Triangle 25mm | -0.071mm (INV7=0.627, MPC=0.556) | 0.675 | 0.282 |
| U-shape | -0.069mm (INV7=0.560, MPC=0.490) | 0.606 | 0.216 |

Under a frozen (stale) Jacobian, MPC is USUALLY far more robust than the
inverse (rectangle: INV-7 degrades 2.6x, MPC only 1.3x; triangle: both
degrade mildly, ~1.2x each) -- but this robustness is NOT universal: on the
U-shape, MPC fails outright (diverges, requires an abort) at one specific
corner while INV-7 degrades but stays bounded. See
`03_ushape/tables/MPC-FJ_excluded_notes.md`.

## What's established vs still open (evidence hierarchy, see rectangle/supplementary)

1. **Actuator authority** (matched authority via INV-7's normalized-DLS
   7th channel) explains a large part of the ORIGINAL (INV-6 vs MPC)
   advantage -- see `01_rectangle/tables/run_level_summary_INV6.md` vs
   `run_level_summary_INV7_matched.md`.
2. **Jacobian staleness** explains the FJ lateral failure of the inverse
   law specifically (direct DLS inversion of a stale J) --
   `01_rectangle/supplementary/jacobian_geometry_drift.txt`.
3. **Constraint clipping** is NOT the dominant mechanism for the inverse's
   FJ failure (median rotation 1.6-2.9deg, wrong-sign correlation with the
   error) -- `01_rectangle/supplementary/clip_angle_diagnostic.txt`.
4. **Disturbance persistence** (beta_d=1) has a large effect on MPC's
   near-term (+3) prediction quality, both under SJ and FJ --
   `01_rectangle/supplementary/beta_disturbance_persistence_ablation.txt`.
   Caveat: this is offline counterfactual evidence (single-tick replay
   against real states), not a measured live RMS effect.
5. **Horizon length**: most of MPC's advantage is present already at N=3
   (the minimum causal horizon at d=2 delay); a smaller, real,
   monotonically-saturating benefit continues to N=10 --
   `01_rectangle/supplementary/horizon_sweep_analysis.txt` and
   `horizon_u0_agreement.png`. NOT YET ISOLATED as a standalone
   "N=3-only QP vs DLS" live comparison.
6. **Insertion feedback specifically** (vs MPC with u_L pinned to
   feedforward) contributes a real but SMALL slice of MPC's advantage
   (~0.04-0.10mm out of ~0.67mm total SJ gap on the rectangle) --
   `01_rectangle/supplementary/mpc_insertion_feedback_ablation_h3.txt` /
   `_h10.txt`.

## Known caveats to carry into the paper

- E_u is dimensionless/normalized (S_u^-1-scaled), NOT millimetres -- label
  it correctly (flagged explicitly during the investigation, now consistent
  across all tables in this package).
- The disturbance-persistence (beta_d) finding is offline-replay evidence,
  not a live closed-loop measurement -- state it as "strong offline
  mechanistic evidence," not a measured effect size.
- Every quantitative table in this package is regenerated directly from the
  saved `*_run_level_summary.pkl` files (single source of truth) via a
  short script, not manually transcribed -- if you need to re-verify or add
  a metric, edit the summary pkl generation, not the tables by hand.
- The 20mm-base triangle and S-curve results are explicitly NOT in this
  package (insufficient n) -- do not resurrect the early n=1-2 numbers into
  the paper without collecting to n=5 first, since the 25mm-base triangle
  result already demonstrated that small-n reads on this system can flip
  sign as more reps come in.
