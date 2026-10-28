# Jacobian-staleness robustness: rectangle vs. wide triangle, raised (+30mm) workspace

**Date:** 2026-09-24. **Data:** today's raised-workspace live campaign only (rectangle: 4 SJ + 4 FJ reps per controller; wide triangle, corrected closed 3-sided geometry: 4 SJ + 4 FJ reps per controller). U-shape excluded — its FJ/SJ grid was not completed today. Figures and the script that built them live alongside this document in `close_loop_logs/myrun/paper_style_analysis_2026-09-24/`.

This reconstructs the analysis style of the existing paper-package figures (`paper_package_2026-09-21/00_main/Figure{1,3,4}_*.png`) but is **not a byte-for-byte reproduction** — that generating script wasn't preserved from the earlier session. Panel definitions that required reconstruction are documented in `paper_style_report.py`'s docstring and repeated below where they matter for interpretation.

---

## Figure 1 — spatial path tracking (`fig1_spatial_path_tracking.png`)

Task-plane x/y trajectory under the genuine, per-tick Scheduled Jacobian (SJ) — every rep shown faint, mean trajectory bold, desired path dashed. This is the "does it work at all" sanity check, not the robustness question.

Both controllers track both shapes closely under SJ. The rectangle shows INV-7 (green) with visibly more high-frequency jitter than MPC (blue) along the straight edges — consistent with INV-7's proportional correction reacting to every measurement noise cycle, while MPC's optimization implicitly smooths through its cost structure. The triangle shows both controllers tracking the corners tightly, with INV-7 again slightly noisier, and both slightly undershooting the apex.

**Takeaway:** under the honest model, neither controller nor shape has a qualitative tracking problem — the interesting behavior only appears once the Jacobian is frozen (Figures 3–4).

---

## Figure 2 — cross-shape SJ performance (`fig2_cross_shape_sj_performance.png`)

Three panels (RMS tracking error, normalized feedback effort `E_u`, joint command-increment constraint activity), INV-7 vs MPC, per shape, SJ only — a clean-model performance comparison, orthogonal to the FJ/SJ robustness question. `E_u` is each rep's mean normalized combined feedback effort (`||S_u^-1(u0-u_ref)||`); "activity" is each rep's mean over all (tick, joint) pairs of `|dq_i|/max_joint_step_rad`, as a percentage (a utilization average, not the `P(active>=0.9)` saturation-probability reported elsewhere in this project, which is far smaller — see the script docstring for why).

| Shape | Controller | RMS (mm) | `E_u` | activity (%) |
|---|---|---|---|---|
| Rectangle | INV-7 | 0.463 ± 0.025 | 0.419 ± 0.028 | 7.76 ± 0.50 |
| Rectangle | MPC | 0.393 ± 0.018 | 0.148 ± 0.008 | 5.21 ± 0.10 |
| Triangle | INV-7 | 0.783 ± 0.197 | 0.472 ± 0.012 | 11.39 ± 0.12 |
| Triangle | MPC | 0.591 ± 0.038 | 0.278 ± 0.016 | 11.43 ± 0.31 |

MPC beats INV-7 on RMS on both shapes while using markedly less normalized effort (2.8× less on the rectangle, 1.7× less on the triangle) — consistent with the efficiency edge already established in the mechanism/effort figures earlier today. The constraint-activity picture is more shape-dependent than the effort picture: on the rectangle MPC's joint increments run at a clearly lower average utilization than INV-7's (5.2% vs 7.8%), but on the triangle the two are statistically indistinguishable (~11.4% each) — MPC's efficiency advantage there comes from spending its effort more usefully (see Figure 4's D1/D2 panels), not from moving the joints less.

---

## Figure 3 — FJ vs SJ robustness (`fig3_fj_sj_robustness.png`)

RMS error, Frozen-J (FJ) vs Scheduled-J (SJ), per controller, per shape. Individual rep RMS shown as faint dots (full-run RMS, including whatever fraction of the path a safety-aborted rep reached — not truncated to a common window, so a rep that aborted early after a moderate excursion can show a *lower* full-RMS than one that ran longer before diverging; this is why MPC-FJ's rectangle dots span 0.90–2.31mm despite a similar underlying failure mode).

| Shape | Controller | FJ RMS (mm) | SJ RMS (mm) | FJ/SJ ratio | FJ completion |
|---|---|---|---|---|---|
| Rectangle | INV-7 | 1.206 ± 0.234 | 0.463 ± 0.025 | 2.6× | 4/4 |
| Rectangle | MPC | 1.733 ± 0.582 | 0.393 ± 0.018 | 4.4× | **1/4** (3 safety-terminated) |
| Triangle | INV-7 | 0.927 ± 0.089 | 0.783 ± 0.197 | 1.2× | 4/4 |
| Triangle | MPC | 0.616 ± 0.013 | 0.591 ± 0.038 | 1.04× | 4/4 |

**This is the headline result, and it is shape-dependent, not a universal MPC property.** On the rectangle, freezing the Jacobian is catastrophic for MPC specifically — 3 of 4 reps safety-terminated (`tcp_out_of_workspace` / `insertion_offset_exceeded`), and the one survivor still ran at 2.3× its SJ RMS. INV-7 degrades on the rectangle too (2.6×) but never diverges. On the triangle, **neither controller shows meaningful FJ degradation** — MPC's FJ/SJ ratio is 1.04×, essentially noise, and INV-7's is a modest 1.2×. The same frozen-Jacobian manipulation produces a qualitatively different outcome depending on the shape being traced.

---

## Figure 4 — mechanism figures (`fig4_mechanism_rectangle.png`, `fig4_mechanism_triangle.png`)

Six panels per shape, all built from the FJ/SJ rep sets above.

**Panels A/B (Scheduled-J / Frozen-J tracking, mean±SD vs path progress):** the rectangle's Panel B is the clearest single picture in this whole analysis — INV-7 (green) stays under ~1mm for the entire path, while MPC (blue) tracks comparably to INV-7 up to ~70% progress and then runs away to >7mm mean (with SD bands reaching into the 10mm+ region) by 100%. The triangle's Panel B shows both controllers drifting upward together late in the path (both reaching ~1.5–2mm near 80–100% progress) with no divergence — a shared, mild degradation rather than a controller-specific failure.

**Panel C (Frozen-J lateral Jacobian rotation):** the angle between the frozen model's lateral (y) row and the genuine schedule's y-row at the same point, overlaid with the INV-7-minus-MPC lateral-error gap. On the rectangle this rotation climbs smoothly to ~50° and stays there — a single, sustained misalignment regime. On the triangle it is far more erratic — repeated spikes to 100–125° with sharp returns to near 0°, reflecting the geometry passing through near-degenerate directions at the triangle's corners. Despite the *rotation itself* reaching higher peak values on the triangle, MPC does not diverge there — consistent with the earlier finding (this session's full FJ diagnostic) that what matters is whether the *usable* singular directions drift, not the raw rotation magnitude; the triangle's spikes appear to be transient/localized rather than a sustained drift in the directions the controllers actually draw correction from.

**Panels D1/D2 (useful lateral corrections / net correction efficiency):** D1 measures the fraction of ticks where the commanded correction's y-component is sign-aligned with reducing the *current* lateral error — this is a **necessary-but-not-sufficient** signal, since it only checks direction under the controller's own (possibly wrong) model, not whether that direction was actually validated by the real beam response. This is why MPC-FJ on the rectangle scores *highest* on D1 (~90% useful) while still diverging: even during the runaway, the QP keeps commanding corrections that its own frozen model says point the right way — it is D2 (net efficiency, which uses the *realized* error change, not the model's belief) that tells the true story. MPC-FJ's D2 on the rectangle is strongly negative (−0.24, with individual reps down to −0.37) and far worse than every other condition, confirming the corrections were confidently wrong, not merely under-strength. On the triangle, D2 for MPC-FJ is mildly negative (−0.009) and comparable to MPC-SJ (−0.007) and INV-7's two conditions — no condition stands out as catastrophically inefficient, matching the absence of divergence.

---

## Synthesis

1. **FJ is worse than SJ, on both shapes, for both controllers** — the direction of the effect is universal (Figure 2's ratios are all ≥1.0).
2. **The *severity* is shape-dependent, not controller-dependent in a fixed way.** On the rectangle, MPC is dramatically more fragile than INV-7 under a stale Jacobian (4.4× vs 2.6× RMS ratio, and only MPC produces outright safety terminations). On the triangle, both controllers are nearly indifferent to the frozen model (1.04× and 1.2×) — the rectangle's MPC-specific fragility mechanism (Panel C's sustained lateral-row rotation feeding a self-reinforcing "confidently wrong" correction loop, quantified by D2's strongly negative net efficiency) simply does not engage on this triangle geometry.
3. **Mechanistically**, the distinguishing factor is not the peak Jacobian misalignment (the triangle's Panel C peaks are larger than the rectangle's) but whether that misalignment is *sustained* in the directions the controller relies on. The rectangle's long straight edges hold the beam in a slowly-drifting configuration for extended stretches, letting a frozen linearization become progressively — and then persistently — wrong. The triangle's sharper, more varied path keeps passing through different local geometries, so a frozen snapshot's error doesn't compound in the same directed way.
4. **Practical implication:** "MPC is more fragile than INV-7 to model staleness" (this session's rectangle-only finding) should not be treated as a general controller property — it is at minimum shape/path-geometry-conditional, and the wide triangle is a direct counterexample at n=4.

## Caveats

- n=4 per condition; the rectangle's MPC-FJ RMS in particular has enormous rep-to-rep variance (0.90–2.31mm) driven by *when* each rep happened to hit its safety boundary, not a stable underlying number — treat the rectangle MPC-FJ mean as a qualitative "usually fails" signal, not a precise RMS estimate.
- D1/D2 and Panel C's "Jacobian-row rotation" are this session's reconstructions of the reference figure's panel intent, not verified reproductions of an original script (which wasn't preserved). The underlying data and RMS/completion numbers in Figure 2 are the load-bearing, directly-measured results; Figure 3 D1–D2 are secondary, interpretive diagnostics.
- U-shape is excluded throughout — its FJ/SJ grid was started today but explicitly stopped by request before completion, so no FJ-vs-SJ comparison exists for it yet.
