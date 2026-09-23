# U-shape (10x15mm, bimaterial) -- MPC-FJ excluded, see notes: run-level summary (n=5 per condition)

**Axis split (SJ, mm):** INV-7 is actually better in x (0.254 vs MPC's
0.319); MPC's advantage is driven entirely by y (0.498 → 0.372, a 25%
reduction). This supports the reading that the remaining law advantage is
primarily lateral/magnetic-steering, not insertion/axial tracking.

**INV7-FJ has high dispersion** — the mean±sd (1.95±1.31mm) makes this
condition look more stable than the underlying distribution really is. Per-run
RMS (mm): [0.839, 1.365, 1.482, 1.558, 4.516]. **Median=1.48mm, range =
[0.84, 4.52]mm** — one run (4.52mm) is a clear outlier; without it the other
4 runs cluster around 1.06–1.56mm. Keep the individual points visible in any
figure using this condition; don't present the mean alone.

| metric | INV7-FJ | INV7-SJ | MPC-SJ |
|---|---|---|---|
| RMS error (mm) | 1.9519 ± 1.3063 | 0.5595 ± 0.0239 | 0.4902 ± 0.0173 |
| x-axis RMS (mm) | 0.3072 ± 0.0144 | 0.2543 ± 0.0107 | 0.3188 ± 0.0165 |
| y-axis RMS (mm) | 1.9195 ± 1.3181 | 0.4983 ± 0.0243 | 0.3723 ± 0.0095 |
| mean predicted +3 error reduction (mm) | -0.0998 ± 0.0591 | -0.0708 ± 0.0209 | 0.0397 ± 0.0102 |
| normalized total effort (dimensionless) | 0.6822 ± 0.0947 | 0.6059 ± 0.0558 | 0.2163 ± 0.0120 |
| fraction of ticks insertion at velocity limit | 0.5397 ± 0.0224 | 0.4647 ± 0.0375 | 0.2699 ± 0.0214 |
| fraction of ticks a joint at rate limit | 0.1491 ± 0.0419 | 0.1724 ± 0.0362 | 0.0009 ± 0.0018 |
| RMS error, corner segments (mm) | 1.0706 ± 0.1833 | 0.6254 ± 0.0629 | 0.5708 ± 0.0724 |
| RMS error, straight segments (mm) | 2.0593 ± 1.4485 | 0.5484 ± 0.0338 | 0.4739 ± 0.0311 |
| corr(predicted, realized) +3 error reduction | 0.2634 ± 0.0473 | 0.1998 ± 0.0564 | 0.4113 ± 0.0609 |
| slope(realized vs predicted) +3 error reduction | 0.4093 ± 0.0944 | 0.2786 ± 0.0759 | 0.6570 ± 0.1213 |
