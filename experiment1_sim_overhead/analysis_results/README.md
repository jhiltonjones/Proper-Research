# Controller Grid Analysis

Experiments discovered: 5

## Main outputs

- `experiment_summary.csv`: one row per experiment.
- `stage_metrics.csv`: one row per frame and prediction stage.
- `channel_metrics.csv`: one row per frame, stage, and input channel.
- `frame_metrics.csv`: one row per feedback frame.
- `per_experiment/`: detailed figures for each run.
- `cross_experiment/`: broad experiment-level comparison figures.
- `controller_comparisons/`: matched inverse-Jacobian, LTI MPC and SQP MPC comparisons.
- `controller_comparison_frame_metrics.csv`: one row per controller and feedback frame.
- `controller_comparison_summary.csv`: matched controller summary metrics.

## Interpretation

- `lin_nl_error_m` measures local linearization error.
- `applied_nonlinear_error_m` measures nonlinear-plan versus executed rollout mismatch.
- `command_utilization_abs` is `abs(u_j) / input_scale_j`.
- `jacobian_authority_scaled_m` is `||B[:,j] input_scale_j||`.
- `contribution_norm_m` is `||B[:,j] u_j||`.
- `tangential_contribution_m` is signed progress along the reference tangent.
- `lateral_contribution_m` is steering contribution normal to the reference tangent.

Heatmap clipping is display-only. CSV values are never clipped.

## Contact versus no-contact comparison

Matched contact/no-contact runs were found. Samples are paired by nearest
reference progress. The script reports source-pose differences because
trajectory differences can confound the Jacobian comparison.

For a strict same-pose comparison, evaluate both Jacobian models at the
same saved `p_start` using your model factory.

## Three-controller comparison

Matched inverse-Jacobian, LTI MPC and SQP MPC runs were grouped
by bend, contact-model setting, plant setting, rollout length and
prediction horizon. Curves use reference progress where available.
Summary statistics are restricted to the common progress interval
when all compared controllers contain usable progress values.

Jacobian authority is a property of B at the visited pose. Selected
tip contribution is ||B[:,j]u_j|| and therefore combines local
authority with the controller's chosen command.
