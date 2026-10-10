# Rectangle (10x15mm): run-level summary (n=5 per condition)

| metric | INV-FJ | INV-SJ | INV7-FJ | INV7-SJ | MPC-FJ | MPC-SJ |
|---|---|---|---|---|---|---|
| RMS error (mm) | 2.2481 ± 0.1440 | 1.2134 ± 0.0267 | 2.0746 ± 0.0125 | 0.7839 ± 0.0414 | 0.7150 ± 0.0270 | 0.5438 ± 0.0158 |
| x-axis RMS (mm) | 0.9048 ± 0.0368 | 0.8114 ± 0.0640 | 0.3413 ± 0.0335 | 0.4018 ± 0.0406 | 0.4029 ± 0.0279 | 0.2701 ± 0.0129 |
| y-axis RMS (mm) | 2.0574 ± 0.1483 | 0.8993 ± 0.0424 | 2.0460 ± 0.0136 | 0.6723 ± 0.0351 | 0.5900 ± 0.0286 | 0.4719 ± 0.0144 |
| mean predicted +3 error reduction (mm) | -0.0737 ± 0.0288 | -0.0713 ± 0.0199 | -0.0511 ± 0.0246 | -0.0358 ± 0.0104 | -0.0101 ± 0.0076 | 0.0723 ± 0.0227 |
| normalized total effort (dimensionless) | 0.7687 ± 0.1229 | 0.6289 ± 0.0614 | 0.8265 ± 0.0925 | 0.6247 ± 0.0782 | 0.3255 ± 0.0143 | 0.3692 ± 0.0377 |
| fraction of ticks insertion at velocity limit | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.4855 ± 0.0199 | 0.4474 ± 0.0424 | 0.3809 ± 0.0275 | 0.2277 ± 0.0221 |
| fraction of ticks a joint at rate limit | 0.1904 ± 0.0482 | 0.1433 ± 0.0242 | 0.1898 ± 0.0405 | 0.1630 ± 0.0311 | 0.0000 ± 0.0000 | 0.0022 ± 0.0018 |
| RMS error, corner segments (mm) | 3.2033 ± 0.3110 | 1.1235 ± 0.0881 | 3.0132 ± 0.1188 | 0.7393 ± 0.0217 | 0.7699 ± 0.0386 | 0.4992 ± 0.0219 |
| RMS error, straight segments (mm) | 1.9883 ± 0.1147 | 1.2304 ± 0.0418 | 1.8319 ± 0.0317 | 0.7935 ± 0.0510 | 0.7024 ± 0.0293 | 0.5533 ± 0.0156 |
| corr(predicted, realized) +3 error reduction | -0.0171 ± 0.0339 | 0.2947 ± 0.0565 | 0.0827 ± 0.0241 | 0.2354 ± 0.0271 | 0.1467 ± 0.0255 | 0.4466 ± 0.0154 |
| slope(realized vs predicted) +3 error reduction | -0.0291 ± 0.0556 | 0.3614 ± 0.0719 | 0.1343 ± 0.0438 | 0.2651 ± 0.0433 | 0.1683 ± 0.0386 | 0.5729 ± 0.0633 |

## Factorial contrasts (run-level means, n=5 each)

| metric | Delta_law (MPC-INV) | Delta_schedule (SJ-FJ) | Delta_interaction |
|---|---|---|---|
| RMS error (mm) | -1.1014 | -0.6030 | +0.8636 |
| x-axis RMS (mm) | -0.5217 | -0.1131 | -0.0394 |
| y-axis RMS (mm) | -0.9474 | -0.6381 | +1.0400 |
| mean predicted +3 error reduction (mm) | +0.1036 | +0.0424 | +0.0800 |
| normalized total effort (dimensionless) | -0.3515 | -0.0480 | +0.1835 |
| fraction of ticks insertion at velocity limit | +0.3043 | -0.0766 | -0.1532 |
| fraction of ticks a joint at rate limit | -0.1658 | -0.0224 | +0.0492 |
| RMS error, corner segments (mm) | -1.5288 | -1.1753 | +1.8091 |
| RMS error, straight segments (mm) | -0.9815 | -0.4535 | +0.6088 |
| corr(predicted, realized) +3 error reduction | +0.1578 | +0.3058 | -0.0119 |
| slope(realized vs predicted) +3 error reduction | +0.2045 | +0.3975 | +0.0142 |
