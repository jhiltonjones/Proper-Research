# Figure and table captions (IEEE style — paste into manuscript)

Graphics carry data + concise labels only; captions below give the minimum
needed to read the figure. Causal interpretation and discussion of the
findings belongs in the Results text, not the captions — see each shape's
`tables/*.md` and `MANIFEST.md` for the fuller analysis notes those
interpretive sentences are drawn from.

---

**Fig. 1.** Spatial path tracking under scheduled J. Desired tip path
(black, dashed) and measured tip trajectories for INV-7 (green) and MPC
(blue) on the rectangle, triangle, and U-shape paths. Thin lines show
individual runs (n=5); thick lines show the run mean. Axes are equal-aspect
task-plane coordinates in mm, local to each panel (each shape's reference
start point placed at the origin).

**Fig. 2.** Scheduled-J performance across three path geometries.
Individual points denote live runs; bars show mean ± 95% CI (n=5 per
controller and path). MPC yields lower mean RMS tracking error, lower
normalized feedback effort $E_u$, and substantially less joint-rate-
constraint activity than matched-authority INV-7.

**Fig. 3.** Sensitivity to Jacobian staleness. RMS tracking error for
matched-authority INV-7 and MPC under frozen and scheduled Jacobians.
Points denote individual live runs and markers show mean ± 95% CI (n=5 for
completed conditions). Both attempted U-shape MPC-FJ trials were
safety-terminated before path completion; therefore no complete-path RMS
is reported.

**Fig. 4.** Rectangle mechanism analysis. (A, B) Mean ± SD tip-error norm
versus normalized path progress under scheduled and frozen Jacobians; gray
bands mark corners. (C) Frozen-J rotation of the normalized joint-to-$y$
Jacobian row relative to $J(0)$, overlaid with the INV-7–MPC lateral-error
gap. (D1, D2) Directional reliability for all four controller/schedule
conditions, grouped by schedule (SJ, FJ); individual run points plus
mean ± 95% CI. In (D1, D2), a correction is classified as useful when the
total feedback action reduces predicted lateral-error magnitude
($|e_y|$) at the causal +3 stage relative to the corresponding no-new-
correction baseline; D1 reports the percentage of ticks meeting this
criterion. D2 reports net correction efficiency: the signed aggregate
reduction in $|e_y|$ (harmful corrections counted negatively), divided by
normalized joint-feedback effort.

**Fig. 5.** Offline controller-mechanism ablations on recorded rectangle
trajectories. (A) Same-state counterfactual replay comparing $\beta_d=0$
and $\beta_d=1$; paired lines connect results from the same recorded
trajectory. (B) Horizon sensitivity: cosine agreement of the first joint
command with the $N{=}15$ command, and fraction of useful lateral
corrections, versus prediction horizon. Percentages denote ticks for which
the joint command is oppositely directed relative to $N{=}15$. All
ablations use the same recorded live trajectories and are not additional
live trials.

**Table I.** Primary result: matched-authority INV-7 vs MPC under scheduled
J, mean ± SD RMS tracking error and normalized feedback effort $E_u$
across n=5 runs per condition, all three path geometries.

**Table II.** Frozen-to-scheduled RMS ratio for each controller and
geometry; a ratio near 1 indicates little sensitivity to the stale model.
Ratios are computed from condition mean RMS values. U-shape MPC-FJ is
marked NA because both attempted trials were safety-terminated before
completing the path (see Fig. 3).
