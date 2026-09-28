# How to run the closed-loop vessel MPC

This covers `run_mpc_delay_aware_vessel.py` -- the live, process-isolated,
delay-aware MPC controller (exact Q_N=0, R700, contact-aware or no-contact
Jacobian, d=2 delay compensation) that drives the robot along an offline
vessel plan with real-time correction. Run this AFTER the offline plan is
built and the open-loop feedforward check has already passed -- see
`HOWTO_VESSEL_PLANNING.md` steps 1-5 first. This doc assumes you already
have a plan directory (`plans/<name>/time_parameterized_configuration_path`)
and its lumen file.

## 0. What this controller is

- Runs in a **separate process** from the one holding the RTDE/camera
  connections (`worker_process.py`) -- the QP solve never shares a GIL
  with the hardware I/O, so a slow solve can't freeze the robot
  connection. A 70ms per-tick deadline is enforced; a miss just holds the
  last commanded joints for that tick (never applies a stale command).
- Tracks the **beam tip position**, not just joint targets -- the cost
  uses a Jacobian schedule (`--schedule-cache`) mapping joint/insertion
  state to predicted tip position, either the contact-aware model
  (`--contact`) or the no-contact model (`--no-contact`).
- Three **independent safety layers**, all separate from the optimizer's
  own constraints:
  1. insertion-offset abort (`--insertion-offset-abort-mm`, default 5mm):
     aborts if measured insertion drifts too far from the reference.
  2. magnet-z bounds (`--magnet-rise-limit-mm` / `--magnet-floor-margin-mm`):
     aborts if the magnet's FK-computed z leaves a band around its start.
  3. magnet-exclusion-radius abort (`--magnet-exclusion-tolerance-mm`):
     aborts if the magnet gets too close to the vessel centreline.

  All three are computed from the **measured** joints each tick, after a
  command has already been applied -- they are a last-resort net, not
  something the optimizer can plan around.
- As of 2026-09-28, the magnet-exclusion limit is **also** wired directly
  into the QP as a linearized inequality constraint (see
  `delay_aware_mpc.py`'s `_configure_magnet_exclusion` docstring), so the
  optimizer itself now tries to route around the wall instead of relying
  entirely on layer 3 catching it after the fact. This is **on by
  default**; pass `--disable-magnet-exclusion-in-qp` to fall back to the
  old behaviour (post-hoc monitor only) if you need to A/B against it.

## 1. Prerequisites before every run

- Robot in **Remote Control** mode on the teach pendant (not just powered
  on) -- RTDE control commands are refused otherwise, with an explicit
  error naming the cause.
- Camera connected and the vessel/markers in view.
- The offline plan's open-loop check (`run_open_loop_vessel.py`) has
  already passed for this exact plan.
- Physical insertion length matches the plan's own `L0` (within
  `--insertion-tol-mm`, checked automatically by preflight -- but resetting
  first avoids a wasted preflight failure):

```python
from proper_research.hardware.online.vessel_stage_a.checkpoint_beam_shape_campaign import (
    reset_insertion_to_target,
)
from pathlib import Path
reset_insertion_to_target(25.49, live=True, out_dir=Path("/tmp"))  # target L0 in mm, read from the plan
```

- You need a **magnet-exclusion reference-joints file** -- a saved JSON
  with the empirically validated closest-safe-approach joint state (e.g.
  `vessel_magnet_exclusion_reference_joints_2026-09-24.json`). This is
  used to compute the exclusion radius both the post-hoc monitor and (by
  default) the in-QP constraint use. Reuse an existing one for the same
  vessel setup rather than fabricating a new one -- it must be an
  empirically verified safe pose, not a guess.

## 2. Run it

```bash
python -m proper_research.hardware.online.vessel_stage_a.run_mpc_delay_aware_vessel \
    --plan-dir plans/vessel_live_trimmed6mm_2026-09-28/time_parameterized_configuration_path \
    --out-dir close_loop_logs/myrun \
    --run-name vessel_live_trimmed6mm_mpc_contact \
    --schedule-cache /tmp/vessel_live_trimmed6mm_schedule_contact.npy \
    --lumen-file vessel_lumen_robot_frame_raised3cm_trimmed6mm_2026-09-28.json \
    --insertion-max-mm 65 \
    --magnet-exclusion-reference-joints vessel_magnet_exclusion_reference_joints_2026-09-24.json \
    --contact
```

Swap `--contact` for `--no-contact` to run the no-contact-Jacobian
condition instead (same plan, same schedule cache path convention --
just point `--schedule-cache` at a different `.npy` file, since the
contact and no-contact schedules are numerically different and must not
share a cache file).

### What happens, in order

1. **Schedule build or load** -- if `--schedule-cache` doesn't exist yet,
   this builds a genuine from-model Jacobian schedule (~90-150s, one real
   beam solve per reference sample); if it exists, loads it instantly.
   Build it once per (plan, lumen, contact/no-contact) combination and
   reuse the cache on every subsequent run.
2. **Magnet-exclusion radius computed** from your reference-joints file
   (cheap, no motion) -- printed as both the "true" radius and the
   tolerance-reduced live abort threshold.
3. **Magnet-z bounds computed** from the plan's own start pose (cheap, no
   motion).
4. **In-QP magnet-exclusion schedule built** (cheap, analytic FK Jacobians
   along the reference trajectory) and **worker process spawned + warmed**
   -- this happens BEFORE any hardware connection opens, so the one-time
   cost of spawning a fresh Python interpreter + importing
   numpy/scipy/osqp + one warm-up QP solve doesn't land on your first real
   control tick. Takes a few hundred ms to a couple seconds.
5. **Preflight** (skip with `--skip-preflight` only if you've verified the
   robot is already exactly at the plan's start state) -- resets the robot
   to the plan's start joints via a path that's itself checked against the
   magnet-exclusion radius and z-bounds before any motion starts, and
   verifies the measured insertion matches `L0` within `--insertion-tol-mm`.
6. **Closed-loop control** runs at `common.CONTROL_HZ`, streaming servoJ
   commands via the execution-C accumulator, until the plan completes,
   `--max-control-steps` is hit, or a safety abort fires.

### `--dry-run`

Only suppresses the actual servoJ streaming during the control loop
itself -- it does **not** suppress the preflight reset motion, which runs
unconditionally beforehand. Use `--skip-preflight` too if you want a
genuinely motion-free dry run (e.g. to check the schedule build, worker
spawn, and preflight math without moving the robot at all).

## 3. Reading the output

Each run writes to `<out-dir>/<run-name>_<timestamp>/`:

- `controller_metadata.json` -- the frozen controller spec actually used
  this run (weights, delay_samples, beta_d, horizon, magnet bounds/radius,
  the live planner-to-robot frame registration).
- `path_follow.jsonl` -- one row per control tick: solver status
  (`solver_status`/`solver_success`/`infeasible`), commanded/measured
  state, tracking error, timing. Check `solver_success` is `True`
  throughout a run before trusting its tracking numbers.
- `predicted_beam_positions.jsonl` -- the MPC's own predicted tip position
  each tick, useful for post-hoc prediction-vs-measurement comparisons.
- Final console output / a run summary include `stop_reason`
  (`path_complete` on success; `magnet_exclusion_violated`,
  `insertion_offset_violated`, `magnet_z_violated`, or a deadline/worker
  issue otherwise) and the final/max tracking error in mm.

A run stopped early by any of the three post-hoc safety monitors is not a
QP failure -- check `path_follow.jsonl`'s `solver_status` column for the
ticks leading up to the stop; if every one says `ok`, the optimizer found
a valid solution right up to the abort and the wall/insertion/z limit is
what actually stopped the run, not a solver problem.

## 4. Common flags reference

| Flag | Default | Meaning |
|---|---|---|
| `--plan-dir` | required | offline plan's `time_parameterized_configuration_path` dir |
| `--out-dir` / `--run-name` | required / `mpc_delay_aware_vessel_accumC` | where logs go |
| `--schedule-cache` | required | `.npy` path; built once, reused after |
| `--lumen-file` | required | vessel centreline/radius JSON |
| `--insertion-max-mm` | required | must match the value the plan was built with |
| `--magnet-exclusion-reference-joints` | required | empirically-safe closest-approach pose JSON |
| `--contact` / `--no-contact` | required (mutually exclusive) | which Jacobian model the schedule/tracking use |
| `--horizon` | 15 | MPC prediction horizon N |
| `--max-control-steps` | 800 | hard cap on control ticks |
| `--deadline-ms` | 70.0 | per-tick solve deadline before a miss is declared |
| `--insertion-offset-abort-mm` | 5.0 | post-hoc monitor threshold |
| `--insertion-tol-mm` | 3.0 | preflight insertion-match tolerance |
| `--magnet-rise-limit-mm` / `--magnet-floor-margin-mm` | 40.0 / 40.0 | post-hoc magnet-z monitor band |
| `--magnet-exclusion-tolerance-mm` | 5.0 | subtracted from the computed true radius for the post-hoc abort threshold |
| `--disable-magnet-exclusion-in-qp` | off | opt out of the in-QP wall constraint (post-hoc monitor stays on regardless) |
| `--skip-preflight` | off | skip the automatic reset-to-start + insertion check |
| `--dry-run` | off | suppress servoJ streaming only (preflight motion still runs) |

## 5. Running the contact vs no-contact vs frozen-Jacobian comparison

`run_jacobian_schedule_experiments.sh` in this directory is a heavily
commented, step-by-step reference script for the three-way comparison used
throughout the model-necessity study: scheduled contact-aware Jacobian,
scheduled no-contact Jacobian, and a frozen (start-of-path-only) Jacobian.
It resets insertion before every run (required -- the advancer has no
encoder and insertion drifts between runs), uses a separate schedule cache
file per condition, and builds the frozen schedule from the contact
schedule via `np.repeat(sj[0:1], sj.shape[0], axis=0)`. Read it top to
bottom and run one block at a time rather than executing it straight
through -- confirm each run's `stop_reason` before moving to the next.

## 6. If something goes wrong mid-run

- **Preflight refuses the reset move**: the straight-line joint-space path
  to the plan's start would cross the magnet-exclusion radius or z-bounds.
  This is routine when the robot is resting far from the plan's start
  pose, not a fault -- see `run_open_loop_vessel.py`'s
  `_safe_reset_with_retreat` for the retreat-waypoint pattern already used
  elsewhere in this package if you need to replicate it here.
- **`RuntimeError: Command is not allowed due to safety reasons, please
  switch robot to Remote Control mode`**: physical teach-pendant setting,
  not fixable from code -- switch modes on the pendant and rerun.
- **Insertion mismatch at preflight**: reset the physical insertion with
  `reset_insertion_to_target(...)` (see step 1) and rerun.
- **A safety abort mid-run**: check `path_follow.jsonl`'s `solver_status`
  column first (see section 3) to confirm whether the QP was ever
  genuinely infeasible (rare) versus a fully valid solution that hit an
  externally-monitored limit (the common case, especially with
  `--disable-magnet-exclusion-in-qp` set or a stale/frozen Jacobian
  schedule).
