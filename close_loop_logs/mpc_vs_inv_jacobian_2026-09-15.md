# MPC vs. inverse-Jacobian — bimaterial model, 2026-09-15

All runs: bimaterial Cosserat beam model (0.2mm nitinol wire + 40mm PDMS/iron-particle
composite tip, 5mm overlap), feedforward mode, MPC horizon=15, `state_tracking_weight=0.001`
fix active (confirmed via the standard override log line on every run). n=3 unless noted.
RMS/max over the transit phase only; hold = mean error during the terminal hold; final =
error at the last control step. Final/best-known results only — see each shape's own
analysis doc for the intermediate runs that led here.

## Rectangle (10x15mm, closed)
| Controller | RMS (mm) | Max (mm) | Hold (mm) | Final (mm) |
|---|---|---|---|---|
| Inverse-Jacobian+FF | 1.18±0.11 | 2.77±0.24 | 0.46±0.25 | 0.31±0.23 |
| MPC-LTI+FF | **1.04±0.08** | **2.62±0.21** | **0.31±0.06** | 0.32±0.09 |
| MPC-LTV+FF | 1.08±0.04 | 2.60±0.02 | 0.19±0.07 | **0.11±0.03** |

MPC beats inv-Jacobian on every metric. Open-loop baseline: 1.51mm RMS / 3.09mm max / 1.11mm final.

## Triangle, 20mm base (closed)
| Controller | RMS (mm) | Max (mm) | Hold (mm) | Final (mm) |
|---|---|---|---|---|
| Inverse-Jacobian+FF | 1.00±0.03 | 2.23±0.15 | 0.24±0.10 | 0.26±0.18 |
| MPC-LTI+FF | 0.95±0.02 | 2.26±0.20 | **0.15±0.01** | 0.22±0.04 |
| MPC-LTV+FF | **0.93±0.03** | **2.19±0.04** | 0.20±0.08 | 0.31±0.14 |

MPC-LTV wins RMS/max; MPC-LTI wins hold; inv-Jacobian is never best on any metric. Open-loop baseline: 1.24mm RMS / 2.97mm max / 0.86mm final.

## U-shape, 10x15mm (open, non-closed) — at the correct camera exposure (25)
| Controller | RMS (mm) | Max (mm) | Hold (mm) | Final (mm) |
|---|---|---|---|---|
| Inverse-Jacobian+FF | 1.28±0.01 | 2.83±0.06 | 0.21±0.09 | 0.20±0.11 |
| MPC-LTI+FF | **1.09±0.04** | **2.61±0.04** | 0.33±0.09 | 0.36±0.21 |
| MPC-LTV+FF | 1.14±0.02 | 2.71±0.11 | **0.17±0.03** | **0.16±0.08** |

MPC beats inv-Jacobian on RMS/max; MPC-LTV matches or beats it on hold/final too.

**Root cause of the earlier (wrong-exposure) inconsistency:** at the camera's prior exposure
setting, MPC's hold-phase performance was real but degraded (inv-Jacobian won hold/final on
every rep). Traced to a genuine code gap in `close_loop_path_follow.py`: during the terminal
hold, the joint-correction target is recomputed from a **frozen** plan reference each tick
with no accumulator (unlike insertion, which has one) — `q_target = ref_state[frozen] +
scale*delta_q`, capped at 0.006 rad/tick with no `+=`. A large residual entering hold (this
shape's endpoint is a novel terminus never visited earlier in the path, unlike the closed
rectangle/triangle where hold = the already-validated start point) couldn't be closed within
the fixed 2.1s hold window, so error oscillated around an elevated plateau instead of
converging. Correcting the camera exposure to 25 closed most of this gap; the residual
underlying code gap (no bounded accumulated trim for joint corrections during hold) has not
been fixed and likely still contributes on shapes/conditions that leave a larger residual at
hold entry. A separate attempted fix (`--ff-trim-base-switch-at-hold`, switching the hold
anchor to live measured position) was tested and **caused a 6-9mm regression** for every
controller — confirmed not viable, do not use. Full investigation, including the failed fix
and the exposure-vs-accuracy trade-off (worse open-loop RMS at exposure 25):
`close_loop_logs/ushape10x15mm_bimaterial_2026-09-15/FINAL_COMPARISON.md`.

## Bottom line
MPC (LTI and/or LTV, depending on shape) beats inverse-Jacobian on RMS and max on every shape
tested today (rectangle, triangle 20mm, U-shape), and on hold/final too once the U-shape's
camera-exposure and hold-phase issues are accounted for. The `state_tracking_weight=0.001` +
`horizon=15` fixes are the precondition for all of this — without them MPC was worse than open
loop (see day2's `close_loop_logs/day2_triangle20mm_2026-09-12/FINAL_COMPARISON.md`).
