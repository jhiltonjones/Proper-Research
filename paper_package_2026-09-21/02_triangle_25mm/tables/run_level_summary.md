# Triangle (25mm base, bimaterial): run-level summary (n=5 per condition)

| metric | INV7-FJ | INV7-SJ | MPC-FJ | MPC-SJ |
|---|---|---|---|---|
| RMS error (mm) | 0.7506 ± 0.0402 | 0.6235 ± 0.0594 | 0.5798 ± 0.0388 | 0.5561 ± 0.0356 |
| x-axis RMS (mm) | 0.4324 ± 0.0240 | 0.4262 ± 0.0330 | 0.4022 ± 0.0281 | 0.3694 ± 0.0260 |
| y-axis RMS (mm) | 0.6132 ± 0.0381 | 0.4545 ± 0.0550 | 0.4159 ± 0.0457 | 0.4153 ± 0.0293 |
| mean predicted +3 error reduction (mm) | -0.0156 ± 0.0121 | -0.0253 ± 0.0074 | -0.0908 ± 0.0059 | -0.0563 ± 0.0077 |
| normalized total effort (dimensionless) | 0.6649 ± 0.0369 | 0.6754 ± 0.0269 | 0.2628 ± 0.0182 | 0.2823 ± 0.0124 |
| fraction of ticks insertion at velocity limit | 0.5235 ± 0.0295 | 0.4902 ± 0.0339 | 0.3078 ± 0.0076 | 0.2969 ± 0.0215 |
| fraction of ticks a joint at rate limit | 0.1176 ± 0.0390 | 0.1652 ± 0.0195 | 0.0020 ± 0.0027 | 0.0081 ± 0.0027 |
| RMS error, corner segments (mm) | 0.6120 ± 0.0583 | 0.5773 ± 0.0175 | 0.6204 ± 0.0404 | 0.4705 ± 0.0406 |
| RMS error, straight segments (mm) | 0.7658 ± 0.0395 | 0.6282 ± 0.0659 | 0.5744 ± 0.0456 | 0.5650 ± 0.0406 |
| corr(predicted, realized) +3 error reduction | 0.3405 ± 0.0301 | 0.4325 ± 0.0209 | 0.1316 ± 0.0232 | 0.2623 ± 0.0819 |
| slope(realized vs predicted) +3 error reduction | 0.3801 ± 0.0451 | 0.5037 ± 0.0328 | 0.1095 ± 0.0222 | 0.2412 ± 0.0695 |

## Factorial contrasts (run-level means, n=5 each)

| metric | Delta_law (MPC-INV, factorial avg.) | Delta_schedule (SJ-FJ, factorial avg.) | Delta_interaction |
|---|---|---|---|
| RMS error (mm) | -0.1191 | -0.0754 | +0.1033 |
| x-axis RMS (mm) | -0.0435 | -0.0195 | -0.0266 |
| y-axis RMS (mm) | -0.1182 | -0.0797 | +0.1581 |
| mean predicted +3 error reduction (mm) | -0.0531 | +0.0124 | +0.0442 |
| normalized total effort (dimensionless) | -0.3976 | +0.0150 | +0.0091 |
| fraction of ticks insertion at velocity limit | -0.2046 | -0.0221 | +0.0224 |
| fraction of ticks a joint at rate limit | -0.1363 | +0.0268 | -0.0414 |
| RMS error, corner segments (mm) | -0.0492 | -0.0923 | -0.1152 |
| RMS error, straight segments (mm) | -0.1273 | -0.0735 | +0.1282 |
| corr(predicted, realized) +3 error reduction | -0.1896 | +0.1114 | +0.0388 |
| slope(realized vs predicted) +3 error reduction | -0.2666 | +0.1276 | +0.0080 |

**Interaction-sign interpretation (corrected 2026-09-22 — a prior draft of
this write-up had this backwards):** for RMS error, `Delta_interaction =
(MPC_SJ - MPC_FJ) - (INV7_SJ - INV7_FJ) = +0.1033`. The positive sign means
**scheduling helps INV-7 more than it helps MPC**, not the reverse — INV-7's
own schedule effect (0.7506→0.6235mm, a 0.1271mm improvement) is more than
5x larger than MPC's (0.5798→0.5561mm, a 0.0237mm improvement). MPC is
comparatively insensitive to scheduling on this shape; INV-7 benefits more
from the scheduled Jacobian. This is the same direction as the rectangle
(where MPC is also more schedule-robust), just a much weaker effect here.

**Degradation ratio under FJ (RMS error, FJ/SJ):** INV-7 degrades
0.7506/0.6235 ≈ **1.20x**; MPC degrades only 0.5798/0.5561 ≈ **1.04x**.
These are NOT "both ~1.2x" — MPC is close to flat across the schedule
condition on this shape (see MANIFEST.md, corrected from an earlier draft).

## Simple effects (recommended primary comparison, RMS error mm)

| effect | value |
|---|---|
| Delta_law\|SJ = MPC_SJ − INV7_SJ | **−0.0674 mm** |
| Delta_law\|FJ = MPC_FJ − INV7_FJ | −0.1708 mm |
| Delta_schedule\|INV7 = INV7_SJ − INV7_FJ | −0.1271 mm |
| Delta_schedule\|MPC = MPC_SJ − MPC_FJ | −0.0237 mm |

MPC-SJ is descriptively better than INV7-SJ (0.556 vs 0.624mm) but this is a
much smaller effect than on the rectangle (−0.067 vs −0.240mm), and with
n=5 the per-condition 95% CIs visibly overlap — do not oversell this
statistically. If the trial structure supports paired/block-aware inference
(e.g. same-order reps across conditions), use that rather than treating the
10 runs as independent groups.
