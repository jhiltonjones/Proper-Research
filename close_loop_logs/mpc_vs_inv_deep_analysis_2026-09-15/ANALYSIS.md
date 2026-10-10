# Why does MPC beat inverse-Jacobian? Deep analysis across the 3 final-result shapes (2026-09-15)

Covers the three "final result" n=3 comparisons where MPC beat inverse-Jacobian: rectangle
10x15mm, triangle 20mm, and the U-shape at the corrected camera exposure (25). All bimaterial
model, horizon=15, `state_tracking_weight=0.001`, feedforward mode.

## Part 1 — Control theory, grounded in this system
The beam's tip is controlled through a 3x7 Jacobian (d(tip_xyz)/d[6 joints, insertion]) that
is badly rank-deficient in practice: every frozen-Jacobian SVD computed today shows two
well-conditioned directions (sigma1≈0.106, sigma2≈0.044 mm/rad) and one near-null third
direction (sigma3≈1.5e-6 mm/rad, five orders of magnitude weaker) — condition numbers
essentially identical across shapes (rectangle kappa≈69,285, triangle20mm kappa≈69,530,
U-shape kappa≈69,142, since all three plans start from the same 25mm-insertion pose).
Inverse-Jacobian (DLS) is a single-step reactive law with zero model of where the reference
is heading. MPC optimizes a cost over a 15-step receding horizon using a linearized model —
LTI freezes it at sample 0, LTV relinearizes every sample from a cached schedule. The
theoretical prediction ("horizon-based anticipation should help most where the reference
curves hardest") is non-obvious here because the system's rank-deficiency means an
"anticipated" correction can point mostly along the useless near-null direction — this needed
checking against real data, not assuming.

## Part 2 — Per-shape findings

**Rectangle 10x15mm** — RMS: inv 1.18±0.11mm, **LTI 1.04±0.08mm** (best), LTV 1.08±0.04mm.
Jacobian mismatch grows to 158% of its start value along the path, yet LTV (the
relinearizing variant) does NOT win RMS — it wins hold/final instead (0.19/0.11mm vs LTI's
0.31/0.32mm). Mechanism-localization gap is distributed, not concentrated (mean
+0.08mm/bin). Fingerprint: MPC saturates 64-75% of ticks vs inv's 26%, is jerkier, routes
44-54% of command through the exact nullspace vs inv's 0.5%.

**Triangle 20mm** — RMS: inv 1.00±0.03mm, LTI 0.95±0.02mm, **LTV 0.93±0.03mm** (best).
Jacobian evolution is the most severe of the three: condition number up to 5.37M, mismatch
reaches 199%. LTV wins here, consistent with theory (relinearization pays off when the model
drifts the most). Corners auto-detected at samples 81/144 — matching day2's
independently-derived finding on the original non-bimaterial 20mm triangle (~80/144) almost
exactly, a genuine cross-validation of the corner-detection method. Mean localization gap:
+0.04mm/bin — the smallest of the three, despite the worst Jacobian drift. Fingerprint: same
negative-result pattern (59-66% saturation vs 20%, ~3x jerkier, 53-61% nullspace usage vs
0.16%).

**U-shape 10x15mm (exposure=25)** — RMS: inv 1.28±0.01mm, **LTI 1.09±0.04mm** (best), LTV
1.14±0.02mm. Jacobian evolution is the mildest (mismatch only 157%). LTI wins again, matching
rectangle. Mean localization gap: +0.18mm/bin — the LARGEST of the three, despite the mildest
drift. Fingerprint: same pattern (59-61% saturation vs 34%, >2x jerkier, 45-49% nullspace vs
0.67%).

## Part 3 — Synthesis

**Real correlation:** ranking shapes by Jacobian-mismatch severity (triangle20mm 199% >
rectangle 158% > U-shape 157%) predicts which MPC variant wins RMS exactly (LTV wins the
worst-drift shape; LTI wins both lower-drift shapes). Only 3 data points, and rectangle/U-shape's
mismatch values (158% vs 157%) are close enough that this could be coincidence — but it's
directionally consistent with theory.

**Genuine complication:** the SIZE of MPC's advantage does NOT track drift severity —
triangle20mm has the worst drift but the smallest edge (+0.04mm/bin); U-shape has the mildest
drift but the largest edge (+0.18mm/bin). Drift severity explains which variant wins, not how
much MPC wins by. That question remains open, same as day2's original "plausible, not
confirmed" conclusion.

**Localization is real but not always concentrated:** all three shapes show a positive average
MPC edge, but only some show a clean single-region concentration (the earlier 8mm-triangle doc
found one; rectangle's is distributed/alternating-sign).

**The most consistent, most counter-intuitive finding:** MPC's behavioral fingerprint is worse
on every metric, on every shape, and it still wins. MPC saturates 2-3x more often (59-66% vs
20-34%), is 2-4x jerkier, and routes 45-61% of its command through directions with zero effect
on the tip (vs inv's <1%) — on all three shapes. The two intuitive "why MPC wins" stories
(better constraint anticipation, smoother commands) are directly contradicted by this data
every time. This replicates day2's original honest negative result across a different beam
model and three new shapes.

## What this doesn't establish
The precise command-level mechanism behind MPC's RMS edge magnitude is unconfirmed (needs
per-tick row/null-space projection correlated with reference curvature — not done here or in
prior work). The LTI-vs-LTV/mismatch correlation is suggestive with only 3 points. Nothing
here explains why the shapes differ so much in Jacobian-drift severity in the first place.

## Data
Figures and results.json for each shape:
- `close_loop_logs/rectangle10x15mm_bimaterial_analysis_2026-09-15/figures/`
- `close_loop_logs/triangle20mm_bimaterial_analysis_2026-09-15/figures/`
- `close_loop_logs/ushape10x15mm_bimaterial_analysis_2026-09-15/figures/`
- `close_loop_logs/mpc_vs_inv_deep_analysis_2026-09-15/figures/` (cross-shape severity chart)

Each shape's `figures/` contains: fig1 tip paths, fig2 error-vs-time, fig3 summary bars, fig4
Jacobian SVD, fig5 Jacobian evolution along path, fig6 Jacobian mismatch vs error, fig7
mechanism localization along path progress, fig8 controller fingerprint (saturation/jerk/nullspace).

## Part 4 — Why does MPC win despite the worse fingerprint?

Part 3 left an unresolved puzzle: MPC saturates 2-3x more often, is 2-4x jerkier, and routes
45-61% of its command through the exact nullspace of the Jacobian (zero tip effect) on all
three shapes — and still wins RMS every time. This section decomposes each controller's raw
per-tick joint correction (`u0_correction[:6]`, rad/s) into a row-space component (the part
that actually moves the tip, `J+ @ J @ corr`, using each tick's own local Jacobian — frozen
for inv/LTI, the relinearised schedule entry at that tick's `ref_index` for LTV) and a
null-space component (the rest), on the same three median-RMS representative reps used
throughout this analysis. Four specific hypotheses were tested.

### H1 — Absolute row-space magnitude: SUPPORTED, strong and consistent
| Shape | inv row-space (rad/s) | MPC-LTI row-space (rad/s) | MPC-LTV row-space (rad/s) |
|---|---|---|---|
| Rectangle | 0.057 | 0.111 (1.9x) | 0.121 (2.1x) |
| Triangle 20mm | 0.049 | 0.098 (2.0x) | 0.088 (1.8x) |
| U-shape (exp=25) | 0.059 | 0.107 (1.8x) | 0.114 (1.9x) |

MPC's *effective* command — after subtracting out everything that goes into the useless null
space — is still roughly **2x larger in absolute magnitude** than inverse-Jacobian's, on
every shape, both MPC variants. This is not a fraction-of-a-bigger-total artifact: it holds
in raw rad/s terms.

### H3 — DLS damping conservatism: SUPPORTED (same evidence as H1)
Inverse-Jacobian's damped-least-squares regularization is doing more than just avoiding the
null space (it's 99.9-100% row-space-aligned on every shape, essentially perfect in that
narrow sense) — it is also making the controller genuinely **cautious in the row space
itself**. inv-Jacobian's row-space command sits at 0.049-0.059 rad/s throughout; MPC's sits
at 0.088-0.121 rad/s despite wasting 14-27% of its own total magnitude in the null space.
Net effect: MPC ends up *more* aggressive where it actually counts, not less, even though it
is visibly "messier." This is the most consistent, best-supported finding in this section and
the most direct answer to "why does MPC win despite the worse fingerprint": inverse-Jacobian's
extra discipline costs it real authority, and that cost outweighs the benefit of never wasting
command in the null space.

### H2 — Directional efficiency (reactive vs. predictive alignment): MIXED, not a clean win for "MPC anticipates"
For each tick, the row-space command's Cartesian effect (`J @ row_comp`) was compared against
the direction to the CURRENT reference point and the direction to the reference point 15 ticks
ahead (matching MPC's horizon).

| Shape | inv (react / pred / gap) | MPC-LTI (react / pred / gap) | MPC-LTV (react / pred / gap) |
|---|---|---|---|
| Rectangle | 0.91 / 0.45 / **-0.46** | 0.64 / 0.28 / -0.36 | 0.74 / 0.41 / -0.34 |
| Triangle 20mm | 0.89 / 0.51 / **-0.39** | 0.50 / 0.27 / -0.23 | 0.70 / 0.42 / -0.27 |
| U-shape (exp=25) | 0.90 / 0.43 / **-0.47** | 0.63 / 0.27 / -0.36 | 0.82 / 0.48 / -0.34 |

Two things are true simultaneously, and they point in different directions:
- **Relative drop-off is smaller for MPC on every shape** (a consistently less negative gap)
  — i.e. MPC's alignment degrades less than inv's when you compare "aimed at right now" vs.
  "aimed at 15 ticks ahead." Taken alone this supports the anticipation story.
- **But inv's raw alignment with the current target (0.89-0.91) is far higher than MPC's
  (0.50-0.82)**, and inv's raw alignment with the FUTURE target (0.43-0.51) is comparable to
  or higher than MPC's (0.27-0.48) in every single case. inv-Jacobian is not failing to
  anticipate the future — it's just extremely tightly locked onto the instantaneous target
  (exactly what a pure proportional law should do), so it has more to lose when compared
  against a point 15 ticks away. MPC's command is less tightly locked to the instantaneous
  target to begin with, so the comparison is more forgiving, not because MPC is aiming
  further ahead in any absolute sense.

**Verdict: H2 does not hold up as "MPC's command points where the beam needs to go next."**
The smaller relative gap is real and replicates on all three shapes, but it's better described
as "MPC is less reactive" than "MPC is more predictive" — a real distinction the raw numbers
don't support calling anticipation.

### H4 — Does the row-space advantage concentrate where the RMS gap concentrates? NOT SUPPORTED
Correlating each shape's per-bin (20 progress bins) RMS-gap-favoring-MPC against the per-bin
row-space-magnitude advantage (MPC minus inv) gives weak, sign-inconsistent correlations:
rectangle r=-0.21, triangle20mm r=+0.56, U-shape r=-0.38. With only 20 noisy bins per shape
this is not a reliable signal either way, but there is no clean, repeated story here the way
there is for H1/H3. The row-space magnitude advantage looks roughly uniform along the path
(consistent with the flat ~2x ratio in H1); the RMS-gap concentration (Part 3) is driven by
something else not captured by this analysis.

### Bottom line for Part 4
The best-supported explanation for "MPC wins despite a worse fingerprint" is **not** that MPC
anticipates the reference better (H2, not supported) or that its authority advantage is
concentrated where it matters most (H4, not supported) — it's simpler and more consistent than
either: **inverse-Jacobian's DLS damping is conservative in the row space, not just the null
space, leaving real tip-moving authority on the table that MPC uses despite also wasting
roughly half its own command on directions with zero tip effect.** MPC's net row-space
authority is ~2x inv-Jacobian's on every shape tested, which is large enough to outweigh the
waste. This is a genuinely different and more specific mechanism than either of the two
stories day2 already ruled out ("better anticipation," "smoother commands") — it is really
about DLS's damping trading away magnitude, not direction.

### New figures (per shape, under `figures/<shape>/`)
- `figA_rowspace_magnitude.png` — row-space vs. null-space command magnitude, all 3 controllers.
- `figB_directional_alignment.png` — reactive vs. predictive (15-tick-ahead) directional alignment.
- `figC_rowspace_vs_progress.png` — row-space command magnitude along path progress (20 bins).
- `part4_rownull_results.json` (in `figures/`) — full numeric results for all three shapes.
