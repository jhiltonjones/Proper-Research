# Layer 4 — MPC tracking

## Verdict

- **WARN** — The offline Jacobian predicts poorly one step ahead. prediction error is 1.67x the tip motion it is trying to predict. That is the case for relinearising online (SQP) rather than trusting the reference-point Jacobian.
- **WARN** — 775 QP solve(s) exceeded the sample period. worst 275.6 ms against 10.0 ms. In simulation this is invisible; on hardware it is a missed deadline. Report worst-case, not mean, solve time when comparing controllers.
- **WARN** — The acceleration limit saturates on q1. the controller is running against a hard bound, so any extra acceleration demand created upstream translates directly into tracking error rather than faster motion.


## Closed-loop tracking

| quantity | tolerance | max | p95 | rms | budget p95 |
|---|---|---|---|---|---|
| tip position | 1.500 mm | 0.6664 mm | 0.3893 mm | 0.1722 mm | 0.260x |
| tip tangent | 90.0 deg | 24.219 deg | 22.428 deg | 11.731 deg | 0.249x |


## Whose error is it?

Total measured tip error contains an error the plan already had. The controller cannot remove it by tracking the plan better — only by using beam feedback, which is exactly what the beam-output variant does.

| component | rms [mm] | max [mm] | share of total rms |
|---|---|---|---|
| planning (plan's own tip error) | 0.01143 | 0.01936 | 6.6% |
| total measured (closed loop) | 0.1722 | 0.6664 | 100% |


## Quality of the affine beam model

The controller predicts the next tip position from `p_nom + J (z - z_ref) + d_hat`, where J was linearised offline at the reference configuration. Comparing that prediction with what the plant actually did is the direct test of whether an offline Jacobian is sufficient.

- One-step prediction error: rms 66.17 um, max 291.71 um.
- Tip motion per step: rms 39.65 um.
- Relative prediction error: 1.669.


## Solver health and real-time feasibility

- QP success 100.0% (0 failures).
- Solve time mean 139.69 ms, p95 253.59 ms, max 275.63 ms against a 10.0 ms sample period.
