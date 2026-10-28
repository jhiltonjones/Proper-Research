# How to run the closed-loop inverse-Jacobian controller (vessel)

This is the naive-controller comparator against `run_mpc_delay_aware_vessel.py`
(see `HOWTO_CLOSED_LOOP_MPC.md`): one-step resolved-rate feedback instead of a
finite-horizon QP, built on the SAME offline Jacobian schedule MPC uses (not a
live per-tick re-evaluation -- that was tried and found far too slow for
`control_hz=10`, see `run_inverse_jacobian_online_vessel.py`'s module
docstring for the numbers). Isolating the control law as the only variable
between the two runs is the whole point of this comparator, so it must share
the MPC run's `--schedule-cache` for a given plan, not build its own.

There are two scripts. Use `run_inverse_jacobian_online_vessel.py` unless you
specifically need the older one's simpler invocation for a no-contact plan:

| | `run_inverse_jacobian_vessel.py` | `run_inverse_jacobian_online_vessel.py` |
|---|---|---|
| Contact model | No-contact only | `--contact` / `--no-contact` |
| Magnet-exclusion protection | Reset-path check only (one-time, before motion) | Reset-path check **plus** a live in-loop choice: `--magnet-protection {clip,hold,selective,off}` |
| When to use | Quick no-contact comparator run | Everything else -- the current, more complete script |

## Prerequisites

- A plan directory (same one you'd pass to `run_mpc_delay_aware_vessel.py`).
- A pre-built `--schedule-cache` `.npy` for that SAME plan -- **this script
  never builds a schedule itself**, it reuses one. The simplest way to get
  one is to run `run_mpc_delay_aware_vessel.py` against the plan first (it
  builds and caches the schedule as a side effect), then pass that same
  `--schedule-cache` path here.
- Everything `HOWTO_CLOSED_LOOP_MPC.md`'s prerequisites section lists
  (robot/camera connected, OSQP not required here since there's no QP).

## Running: `run_inverse_jacobian_online_vessel.py` (current)

```bash
python -m proper_research.hardware.online.vessel_stage_a.run_inverse_jacobian_online_vessel \
    --plan-dir <plan>/time_parameterized_configuration_path \
    --lumen-file vessel_lumen_robot_frame_2026-10-06_zcorrected.json \
    --insertion-max-mm 110 --contact \
    --schedule-cache <same cache the MPC run for this plan used>.npy \
    --magnet-protection hold \
    --out-dir close_loop_logs/myrun --run-name vessel_invjac_online_contact
```

Flags worth knowing before you pick defaults:

- `--contact` / `--no-contact` -- must match whatever model the plan/schedule
  were built with; mismatching plant and schedule contact-mode confounds the
  comparison with a model error, not a controller-law difference.
- `--damping` (default `0.05`), `--position-gain` (default `0.6`),
  `--nullspace-gain` / `--selective-damping-gain` (default `0.0`) -- the
  resolved-rate controller's own gains, analogous to `inv_2dof_trim`'s
  `kp`/`kn` in the rectangle package.
- `--magnet-protection` (default `clip`) -- **this is the actual experimental
  variable this script exists to test**, not an incidental safety flag (see
  the module's own docstring for the full reasoning):
  - `clip`: the controller anticipatorily projects its own command away from
    the magnet-exclusion/z-workspace limits before it's ever sent -- the same
    linearized half-space the MPC's QP enforces as an inequality, just
    applied as a closed-form projection instead. Makes the naive controller
    look smarter about the constraint than its own math actually is.
  - `hold`: the controller's math stays completely naive (no clip at all);
    `close_loop_path_follow`'s command-safety gate instead HOLDS position
    (zero command) on any tick whose raw output would violate the
    constraint. This is the cleaner demonstration that the naive controller
    does not understand the constraint at all, unlike MPC -- the robot
    visibly stalls at the boundary instead of being quietly corrected.
  - `selective`: `make_magnet_exclusion_selective_clip_gate` -- a narrower
    clip than `clip`'s; see the controller module for exactly what it
    selects.
  - `off`: the raw command goes straight to the robot. **Not a safe hardware
    baseline for this constraint** -- use `hold` to see the naive
    controller's true unprotected behaviour without the physical risk.
- `--beam-base-exclusion-floor-mm` (default `210.0`) / `--start-radius-override-mm`
  -- same reset-path safety floor and radial-start-override mechanism
  `HOWTO_CLOSED_LOOP_MPC.md` documents for the MPC script
  (`common.resolve_start_pose_at_radius`) -- shared machinery, not
  reimplemented here.
- `--magnet-rise-limit-mm` / `--magnet-floor-margin-mm` / `--magnet-protection-margin-mm`
  -- tuning for exactly where the clip/hold/selective boundary sits relative
  to the hard physical limits; see the module docstring for the derivation.
- `--max-tracking-error-mm` (default `5.0`) -- generic abort, same as every
  other script in this package.

What this script does **not** have, by design, not by oversight: no reactive
insertion-offset-drift monitor (that's `ProcessIsolatedDelayAwareAdapter`'s
MPC-only machinery) and no reactive magnet-exclusion monitor from *measured*
joints after a command lands (MPC gets this as a second line of defense; here
`clip`/`hold`/`selective` are each the only line of defense -- `hold`'s gate
*is* the line of defense, not an extra one on top of a clip).

## Running: `run_inverse_jacobian_vessel.py` (older, no-contact only)

```bash
python -m proper_research.hardware.online.vessel_stage_a.run_inverse_jacobian_vessel \
    --plan-dir <plan>/time_parameterized_configuration_path \
    --schedule-cache <cache>.npy \
    --out-dir close_loop_logs/myrun --run-name vessel_invjac_ltv
```

Simpler because it has fewer knobs, not because it's safer: it has no live
in-loop magnet-exclusion protection at all (only the one-time reset-path
check via `--exclusion-floor-mm`, default `210.43`) and no contact-model
choice (no-contact always). If you've already moved on to the newer script's
`--magnet-protection` flags for other runs, use `run_inverse_jacobian_online_vessel.py
--no-contact` here too rather than switching scripts, so every comparison in
one campaign went through the same safety machinery.

## Reading results

Same convention as the MPC script and the rest of this package:
`close_loop_logs/<run-name>/path_follow.jsonl` (per-tick diagnostics) and
`tip_trajectory.csv` (raw tip position/error/joint trace). There is no
predicted-beam-position log here (that's `ProcessIsolatedDelayAwareAdapter`'s
MPC-only `prediction_log_path`) -- the inverse-Jacobian controller has no
internal prediction to log.
