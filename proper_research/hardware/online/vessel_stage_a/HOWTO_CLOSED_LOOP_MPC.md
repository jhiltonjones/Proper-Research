# How to run the closed-loop vessel MPC

This covers `run_mpc_delay_aware_vessel.py` -- the live, process-isolated,
delay-aware MPC controller (exact Q_N=0, R700, contact-aware or no-contact
Jacobian, d=2 delay compensation) that drives the robot along an offline
vessel plan with real-time correction. Run this AFTER the offline plan is
built and the open-loop feedforward check has already passed -- see
`HOWTO_VESSEL_PLANNING.md` steps 1-5 first. This doc assumes you already
have a plan directory (`plans/<name>/time_parameterized_configuration_path`)
and its lumen file.

Section 5 covers running the three Jacobian conditions (contact-aware,
no-contact, frozen) for a mechanism comparison, and section 7 covers how to
analyze what comes out -- from a five-second sanity check up to the
mechanistic replay techniques used in the 2026-09-30 contact-vs-no-contact
ablation study.

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
- Four **independent safety layers**, all separate from the optimizer's
  own constraints:
  1. insertion-offset abort (`--insertion-offset-abort-mm`, default 5mm):
     aborts if measured insertion drifts too far from the reference.
  2. magnet-z bounds (`--magnet-rise-limit-mm` / `--magnet-floor-margin-mm`):
     aborts if the magnet's FK-computed z leaves a band around its start.
  3. magnet-exclusion-radius abort (`--magnet-exclusion-tolerance-mm`):
     aborts if the magnet gets too close to the beam-base pivot point.
  4. a **generic TCP-workspace box** (`cfg.workspace_xyz_min_m`/`max_m` in
     `close_loop_path_follow.py`'s main loop, overridden per-vessel-run near
     the top of this script's `main()`) -- aborts if the flange leaves an
     axis-aligned box. This one is **not wired into the QP at all** (see
     section 7.4) -- it is a pure post-hoc reactive check, tuned by hand
     from observed flange excursion on prior runs. If you change plans/
     insertion depth/exclusion floor substantially, re-check this box
     against your own runs' flange range before trusting it to catch
     anything. As of 2026-10-01 this override is conditional on
     `--z-raise-mm` (`< 10mm` picks an unraised-tuned box, otherwise the
     raised/v4-tuned one) -- it was previously a single hardcoded box
     derived only from the raised setup's own flange range, which would
     have falsely tripped almost immediately on an unraised plan (measured
     flange z there dips below that box's old z_min). If you build a plan
     at a z-raise this doc hasn't seen before, don't trust either hardcoded
     box -- derive your own from an open-loop run's logged joints (see
     `HOWTO_VESSEL_PLANNING.md` section 5) before running closed-loop MPC.

  All four are computed from the **measured/commanded** joints each tick,
  after a command has already been applied -- they are a last-resort net,
  not something the optimizer (except as noted below) can plan around.
- The magnet-exclusion limit (layer 3) is **also** wired directly into the
  QP as a linearized inequality constraint (see `delay_aware_mpc.py`'s
  `_configure_magnet_exclusion` docstring), so the optimizer itself tries
  to route around the exclusion zone instead of relying entirely on layer 3
  catching it after the fact. This is **on by default**; pass
  `--disable-magnet-exclusion-in-qp` to fall back to post-hoc-only.
- **2026-09-30 additions** (see `--zero-input-reference-in-r`,
  `--magnet-exclusion-robust-margin-mm`, `--magnet-exclusion-clearance-gain`
  in section 4): a live root-cause investigation found the QP's hard
  exclusion constraint was never actually wrong about its own commanded
  state, but real servo-tracking error between commanded and measured
  joints ate into the margin unpredictably, and the controller had no
  cost-side incentive to keep any margin in reserve once the hard
  constraint was satisfied. Two independently-toggleable fixes exist now:
  a small robust margin tightening the QP's own hard radius, and a soft
  quadratic-hinge cost that proactively pulls away from the wall before the
  hard constraint would ever bind. See section 5 for how these were
  ablated against each other on live hardware.
- The magnet exclusion itself is a **fixed single point + radius** (the
  beam-base pivot, `--beam-base-exclusion-floor-mm`), not a full
  vessel-centreline distance field -- see `compute_magnet_exclusion`'s
  docstring in this file for the 2026-09-28 rework. `--magnet-exclusion-
  reference-joints` is accepted but ignored (kept only so old invocations
  don't break).

## 1. Prerequisites before every run

- Robot in **Remote Control** mode on the teach pendant (not just powered
  on) -- RTDE control commands are refused otherwise, with an explicit
  error naming the cause.
- Camera connected and the vessel/markers in view.
- The offline plan's open-loop check (`run_open_loop_vessel.py`) has
  already passed for this exact plan.
- Physical insertion length matches the plan's own `L0` (within
  `--insertion-tol-mm`, checked automatically by preflight -- but resetting
  first avoids a wasted preflight failure). The advancer has **no
  encoder**, so insertion never moves on its own but also never
  self-corrects -- reset it before every single run, not just the first
  one of a session:

```python
from proper_research.hardware.online.vessel_stage_a.checkpoint_beam_shape_campaign import (
    reset_insertion_to_target,
)
from pathlib import Path
reset_insertion_to_target(25.38, live=True, out_dir=Path("/tmp"))  # target L0 in mm, read from the plan
```

  (Get the exact `L0` for your plan via
  `common.load_plan_initial_state(plan_dir)` rather than hardcoding it --
  see `run_jacobian_schedule_experiments.sh` for the one-liner.)

## 2. Run it

```bash
python -m proper_research.hardware.online.vessel_stage_a.run_mpc_delay_aware_vessel \
    --plan-dir plans/vessel_lumen_2026-09-29_v4_zraise42/time_parameterized_configuration_path \
    --out-dir close_loop_logs/myrun \
    --run-name my_contact_run \
    --schedule-cache /tmp/vessel_lumen_2026-09-29_v4_schedule_contact.npy \
    --lumen-file vessel_lumen_robot_frame_zraise42_2026-09-29.json \
    --insertion-max-mm 100 \
    --z-raise-mm 42.044008750641574 \
    --beam-base-exclusion-floor-mm 220.0 \
    --insertion-offset-abort-mm 10.0 \
    --zero-input-reference-in-r \
    --magnet-exclusion-robust-margin-mm 2.0 \
    --magnet-exclusion-clearance-gain 50000 \
    --magnet-exclusion-soft-radius-mm 225.0 \
    --contact
```

Swap `--contact` for `--no-contact` to run the no-contact-Jacobian
condition instead (same plan, same schedule-cache convention --
just point `--schedule-cache` at a different `.npy` file, since the
contact and no-contact schedules are numerically different and must not
share a cache file). See section 5 for the frozen-Jacobian condition.

Leave out `--zero-input-reference-in-r --magnet-exclusion-robust-margin-mm
--magnet-exclusion-clearance-gain --magnet-exclusion-soft-radius-mm`
entirely to get the plain baseline controller (no robustness mechanisms) --
they all default to off/0.0.

### What happens, in order

1. **Schedule build or load** -- if `--schedule-cache` doesn't exist yet,
   this builds a genuine from-model Jacobian schedule (contact model: ~20-40
   minutes, one real beam-plus-wall-contact equilibrium solve per reference
   sample; no-contact model: much faster, no contact solve); if it exists,
   loads it instantly. Build it once per (plan, lumen, contact/no-contact)
   combination and reuse the cache on every subsequent run -- do not delete
   it between reps of the same condition.
2. **Magnet-exclusion radius** printed (fixed beam-base point/radius, see
   section 0) -- both the "true" floor and the tolerance-reduced live abort
   threshold, and (if `--magnet-exclusion-robust-margin-mm` is set) the
   separate, slightly tighter QP-only radius.
3. **Magnet-z bounds computed** from the plan's own start pose (cheap, no
   motion).
4. **In-QP magnet-exclusion + z-workspace schedule built** (cheap, analytic
   FK Jacobians along the reference trajectory) and **worker process
   spawned + warmed** -- this happens BEFORE any hardware connection opens,
   so the one-time cost of spawning a fresh Python interpreter + importing
   numpy/scipy/osqp + one warm-up QP solve doesn't land on your first real
   control tick.
5. **Preflight** (skip with `--skip-preflight` only if you've verified the
   robot is already exactly at the plan's start state) -- resets the robot
   to the plan's start joints via a path that's itself checked against the
   magnet-exclusion radius and z-bounds before any motion starts, and
   verifies the measured insertion matches `L0` within `--insertion-tol-mm`.
6. **Closed-loop control** runs at `common.CONTROL_HZ` (10Hz), streaming
   servoJ commands via the execution-C accumulator, until the plan
   completes, `--max-control-steps` is hit, or a safety abort fires.

### `--dry-run`

Only suppresses the actual servoJ streaming during the control loop
itself -- it does **not** suppress the preflight reset motion, which runs
unconditionally beforehand. Use `--skip-preflight` too if you want a
genuinely motion-free dry run (e.g. to check the schedule build, worker
spawn, and preflight math without moving the robot at all).

## 2b. Shifting the tracking centreline toward a wall

Two different things can be meant by "shift the centreline 1mm toward
the right wall," and they are NOT interchangeable -- pick based on what
you're actually testing.

### Option A -- live-only shift (`--right-shift-mm`): fast, single-run, no rebuild

Add `--right-shift-mm 1.0` (or any float; negative shifts left) to the
`run_mpc_delay_aware_vessel.py` command from section 2:

```bash
python -m proper_research.hardware.online.vessel_stage_a.run_mpc_delay_aware_vessel \
    --plan-dir plans/vessel_lumen_2026-09-30_realigned_insmax80/time_parameterized_configuration_path \
    --out-dir close_loop_logs/myrun \
    --run-name my_contact_run_right1mm \
    --schedule-cache /tmp/vessel_lumen_2026-09-30_realigned_insmax80_schedule_contact.npy \
    --lumen-file vessel_lumen_robot_frame.json \
    --insertion-max-mm 80 \
    --z-raise-mm 0.0 \
    --right-shift-mm 1.0 \
    --contact
```

This moves ONLY the live controller's `reference.desired_position_m`
(what the running MPC tracks) sideways by the given amount, computed
from the tangent/normal of the existing reference path -- see
`_apply_right_shift` in `worker_process.py`. It does **not** touch:

- the offline plan (`--plan-dir`'s own time-parameterized path),
- the Jacobian schedule (`--schedule-cache`'s `.npy` contents),
- the wall/contact model used by the contact-aware Jacobian, or
- the magnet-exclusion constraint geometry.

So the robot tracks a target 1mm closer to the wall than the plan was
ever actually solved against, but the contact-aware model's own idea of
"where the wall is" is unchanged -- useful for a single quick test of
sensitivity to the tracking target, cheap because it needs no plan
rebuild and can reuse an existing schedule cache, but it is **not** a
faithful "redo the whole offline optimization against a moved vessel"
experiment. You'll see the applied shift printed by the worker at
startup (`[worker] right-shift applied to the LIVE TRACKING TARGET
only: ...`); confirm the printed max displacement matches what you
expect before trusting the run.

### Option B -- offline-configuration shift: slow, changes the plan + wall model itself

If you want the shift to also affect what Layer 1's optimizer was
actually solved against (i.e. you want the shifted line to be *the*
centreline, not just what the live controller happens to aim at), you
have to shift the lumen file itself and rebuild the plan + both
Jacobian schedules from it. Use
`shift_lumen_centerline.py` (mirrors `trim_vessel_lumen.py`'s structure;
same tangent/normal convention as Option A, applied to every centreline
point instead of just the live reference):

```bash
python -m proper_research.hardware.online.vessel_stage_a.shift_lumen_centerline \
    --lumen-file vessel_lumen_robot_frame.json \
    --shift-mm 1.0 --direction right \
    --z-raise-mm 0.0 \
    --out vessel_lumen_robot_frame_right1mm.json
```

Then rebuild the plan from the shifted file exactly as in
HOWTO_VESSEL_PLANNING.md section 2 (expect the same build time as any
other plan at this insertion-max -- the 2026-10-01 726-sample/80mm plan
took ~1h48m for Layer 1 alone, see that doc's timing note), then
rebuild BOTH Jacobian schedules against the new plan/lumen (see "Forcing
a schedule rebuild" immediately below) before running closed-loop MPC
against it. This is the right choice when you actually moved/re-aligned
the physical vessel (see HOWTO_VESSEL_PLANNING.md section 1b first --
if the vessel moved, re-registering against the camera is a different,
more likely explanation than wanting to deliberately bias the plan), or
when you want the contact model itself (not just the tracking target)
to see the shifted wall.

### Forcing a Jacobian schedule rebuild (contact and no-contact)

`build_or_load_schedule` in `run_mpc_delay_aware_vessel.py` only checks
whether the `--schedule-cache` path **exists** -- it does not hash or
otherwise check the lumen/plan contents, so a stale cache from before a
lumen or plan change will be silently reused with no warning. After
changing the lumen file or rebuilding the plan (Option B above, or any
other plan change), force a rebuild by either deleting the old cache or
(preferred, so you keep the old one for comparison) pointing
`--schedule-cache` at a new path that doesn't exist yet. Do this for
**both** conditions separately -- the contact and no-contact schedules
are numerically different and must never share a cache file:

```bash
# contact condition -- new cache path forces a fresh ~20-40min build
python -m proper_research.hardware.online.vessel_stage_a.run_mpc_delay_aware_vessel \
    --plan-dir plans/<new-plan>/time_parameterized_configuration_path \
    --schedule-cache /tmp/<new-plan>_schedule_contact.npy \
    --lumen-file <new-lumen-file>.json \
    ... --contact --dry-run --skip-preflight   # schedule build happens before any motion

# no-contact condition -- separate cache path, same plan/lumen
python -m proper_research.hardware.online.vessel_stage_a.run_mpc_delay_aware_vessel \
    --plan-dir plans/<new-plan>/time_parameterized_configuration_path \
    --schedule-cache /tmp/<new-plan>_schedule_nocontact.npy \
    --lumen-file <new-lumen-file>.json \
    ... --no-contact --dry-run --skip-preflight
```

`--dry-run --skip-preflight` together make this motion-free -- it builds
and saves the schedule and then exits without moving the robot, so you
can pre-build both caches ahead of time and run the real (non-dry-run)
reps afterward against the now-cached, already-verified schedules.

## 3. Reading the output

Each run writes to `<out-dir>/<run-name>_<timestamp>/`:

- `controller_metadata.json` -- the frozen controller spec actually used
  this run (weights, delay_samples, beta_d, horizon, magnet bounds/radius,
  the live planner-to-robot frame registration, `contact: true/false`).
  Always check `planner_to_live_t_fit_norm_m` here is 0 (or tiny) --
  a large value means the planner-to-live frame registration picked up an
  unexpected offset (see section 6's frame-registration note).
- `summary.json` -- the one file to read FIRST for a quick pass/fail check:
  `stop_reason` (`path_complete` on success; `tcp_out_of_workspace([x,y,z])`,
  `magnet_exclusion_violated`, `insertion_offset_violated`,
  `magnet_z_violated`, or a deadline/worker issue otherwise),
  `control_steps`/`reference_samples`, `max_error_mm`, `final_error_mm`.
- `path_follow.jsonl` -- one row per control tick: solver status, commanded/
  measured state, tracking error, timing.
- `predicted_beam_positions.jsonl` -- the per-tick record of record for
  deeper analysis (see section 7): `z_meas`/`q_cmd_k`/`q_cmd_km1`/
  `insertion_cmd_m` (commanded state), `u0`/`u_prev` (solved/previous
  input), `measured_beam_position_m` (vision tip measurement),
  `predicted_beam_positions_m`/`predicted_states` (the MPC's own horizon
  forecast), `jacobian_used` (the exact 3x7 Jacobian the QP solved with
  that tick), `magnet_xyz_m`, `dual_y` (QP dual variables, one per
  constraint row), `qp_objective`/`qp_status`/`qp_iterations`/
  `qp_solve_time_s`/`qp_primal_residual`/`qp_dual_residual`.
- `tip_trajectory.csv` -- tip tracking error time series, easiest format
  for a quick plot.
- `path_follow_plot.png` -- auto-generated tracking-error plot.
- `frozen_jacobian.json` -- a diagnostic dump of the Jacobian schedule used
  (NOT related to the "frozen Jacobian" experimental condition in section
  5 -- this file is written for every run, contact/no-contact/frozen alike;
  it just means "the Jacobian this run's schedule was frozen/serialized
  into for inspection").

A run stopped early by any of the four safety monitors is not necessarily a
QP failure -- check `predicted_beam_positions.jsonl`'s `qp_status` column
for the ticks leading up to the stop; if every one says `solved`, the
optimizer found a valid solution right up to the abort and the safety
monitor is what actually stopped the run, not a solver problem. This was
true for every trip observed in the 2026-09-30 study (see section 7.4).

## 4. Common flags reference

| Flag | Default | Meaning |
|---|---|---|
| `--plan-dir` | required | offline plan's `time_parameterized_configuration_path` dir |
| `--out-dir` / `--run-name` | required / `mpc_delay_aware_vessel_accumC` | where logs go |
| `--schedule-cache` | required | `.npy` path; built once, reused after |
| `--lumen-file` | required | vessel centreline/radius JSON |
| `--insertion-max-mm` | required | must match the value the plan was built with |
| `--z-raise-mm` | 0.0 | z-offset the plan/lumen were built with relative to the unraised setup |
| `--beam-base-exclusion-floor-mm` | 210.43 | true magnet-to-beam-base exclusion floor (before tolerance is subtracted) |
| `--contact` / `--no-contact` | required (mutually exclusive) | which Jacobian model the schedule/tracking use |
| `--horizon` | 15 | MPC prediction horizon N |
| `--max-control-steps` | 800 | hard cap on control ticks |
| `--deadline-ms` | 70.0 | per-tick solve deadline before a miss is declared |
| `--solver-time-limit-s` | 0.03 | OSQP's own wall-clock cutoff per solve (0 disables) |
| `--zero-input-reference-in-r` | off | zero the offline plan's v_ref out of the R cost term (v^T R v instead of (v-v_ref)^T R (v-v_ref)) -- lets the optimizer resolve the redundant DOF from tracking+constraints alone instead of reproducing the offline plan's own choice; see section 0 |
| `--right-shift-mm` | 0.0 | shift ONLY the live tracking target sideways (positive = toward the right wall, negative = left); does not touch the plan/schedule/wall model -- see section 2b for this vs. the offline-rebuild alternative |
| `--insertion-offset-abort-mm` | 5.0 | post-hoc monitor threshold |
| `--insertion-tol-mm` | 3.0 | preflight insertion-match tolerance |
| `--magnet-rise-limit-mm` / `--magnet-floor-margin-mm` | (project const) / 40.0 | post-hoc magnet-z monitor band |
| `--magnet-exclusion-tolerance-mm` | 5.0 | subtracted from the computed true radius for the post-hoc abort threshold |
| `--magnet-exclusion-robust-margin-mm` | 0.0 | tightens ONLY the QP's own hard exclusion radius, on top of the post-hoc floor; the post-hoc monitor is unchanged. Start at ~2.0mm as an experimental value (see section 0) |
| `--magnet-exclusion-clearance-gain` | 0.0 | weight for a separate SOFT quadratic-hinge clearance cost, active below `--magnet-exclusion-soft-radius-mm`; 0 disables. Does not touch the hard constraint |
| `--magnet-exclusion-soft-radius-mm` | 225.0 | soft-cost activation radius; must exceed the QP's own hard radius |
| `--disable-magnet-exclusion-in-qp` | off | opt out of BOTH in-QP wall constraints (post-hoc monitors stay on regardless) |
| `--skip-preflight` | off | skip the automatic reset-to-start + insertion check |
| `--dry-run` | off | suppress servoJ streaming only (preflight motion still runs) |

Note: `cfg.workspace_xyz_min_m`/`max_m` (the generic TCP box, section 0
layer 4) is **not** a CLI flag -- it is set directly in this script's
`main()`. If you change plan/insertion depth, check your own runs' flange
xyz range against it (section 7.4 shows how) before trusting it.

## 5. Running the contact vs no-contact vs frozen-Jacobian comparison

`run_jacobian_schedule_experiments.sh` in this directory is a heavily
commented, step-by-step reference script for the three-way comparison
(written against an older plan; update the `PLAN_DIR`/`LUMEN_FILE`/
`INSERTION_MAX_MM` variables at the top for your own plan, and add the
2026-09-30 flags from section 4 if you want the robustness mechanisms).
It resets insertion before every run, uses a separate schedule-cache file
per condition, and builds the **frozen** schedule from the contact schedule
via:

```python
import numpy as np
sj = np.load(SCHEDULE_CONTACT)
fj = np.repeat(sj[0:1], sj.shape[0], axis=0)   # every row == row 0
np.save(SCHEDULE_FROZEN, fj)
```

i.e. "frozen" means the controller keeps re-using the very first reference
sample's linearization for the entire path, instead of re-linearizing along
the reference trajectory (contact-aware or not). Point `--schedule-cache`
at this frozen `.npy` and pass `--contact` (the flag is a no-op here since
the cache already exists and is loaded as-is -- the `.npy` file's contents,
not the flag, determine the Jacobian actually used).

Read the script top to bottom and run one block at a time rather than
executing it straight through -- confirm each run's `stop_reason` (section
3) before moving to the next.

### Ablation methodology used 2026-09-30 (contact vs no-contact, 5 reps each)

To test whether contact-vs-no-contact Jacobian choice changes *outcomes*
(not just tracking RMS), run **N repetitions of the identical command** per
condition (only `--run-name` changes) and compare completion/failure rate
as the primary result, e.g.:

```bash
for rep in 1 2 3 4 5; do
    reset_insertion   # ALWAYS before each rep
    python -m proper_research.hardware.online.vessel_stage_a.run_mpc_delay_aware_vessel \
        --plan-dir plans/vessel_lumen_2026-09-29_v4_zraise42/time_parameterized_configuration_path \
        --out-dir close_loop_logs/myrun \
        --run-name vessel_v4_mpc_contact_rep${rep} \
        --schedule-cache /tmp/vessel_lumen_2026-09-29_v4_schedule_contact.npy \
        --lumen-file vessel_lumen_robot_frame_zraise42_2026-09-29.json \
        --insertion-max-mm 100 --z-raise-mm 42.044008750641574 \
        --beam-base-exclusion-floor-mm 220.0 --insertion-offset-abort-mm 10.0 \
        --zero-input-reference-in-r \
        --magnet-exclusion-robust-margin-mm 2.0 \
        --magnet-exclusion-clearance-gain 50000 --magnet-exclusion-soft-radius-mm 225.0 \
        --contact
done
```

Then swap `--contact` for `--no-contact` and the schedule-cache path for a
second batch of 5. Treat **runs, not control ticks, as your independent
samples** -- 5 vs 5 is a small-N result (a one-sided Fisher exact test on
5/5 vs 2/5 completions gives p≈0.08; don't lean on the statistics alone,
see section 7 for how to build the mechanistic case that makes a small-N
outcome difference credible).

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
  `reset_insertion_to_target(...)` (see step 1) and rerun. Order matters if
  you're also doing an arm reset: retract insertion BEFORE the arm reset,
  since the arm reset's own preflight includes an insertion-length camera
  check that fails if insertion wasn't retracted first.
- **A safety abort mid-run**: check `predicted_beam_positions.jsonl`'s
  `qp_status` column first (section 3) to confirm whether the QP was ever
  genuinely infeasible (rare) versus a fully valid solution that hit an
  externally-monitored limit (the common case).
- **Unexpected `planner_to_live_t_fit_norm_m` offset in
  `controller_metadata.json`**: `run_open_loop_vessel.py`/
  `run_mpc_delay_aware_vessel.py` monkeypatch `pf._fit_planner_to_robot`/
  `pf._find_shape_npz` to avoid a bug where the frame-fit code searches the
  WHOLE `plans/` tree and can silently pick up an unrelated plan's shape
  file (alphabetically first match), injecting a spurious offset. If you
  add a new entry-point script that calls into this planning stack, port
  the same monkeypatch (see either script's top-of-file for the exact
  pattern) or you may get an invisible offset that only shows up as a
  z-shift baked into your logged tip error.
- **RTDE disconnects / `std::bad_alloc` / robot dropped out of Remote
  Control**: transient hardware faults -- re-run `common.check_robot_safe()`
  and re-check actual joint positions before deciding whether a direct
  retry suffices or a manual retreat-waypoint reset is needed. If the robot
  dropped out of Remote Control, it must be re-enabled on the physical
  teach pendant; nothing in software can do this.

## 7. Analyzing your results

### 7.1 Quick single-run check (30 seconds)

```python
import json
d = json.load(open("close_loop_logs/myrun/<run_dir>/summary.json"))
print(d["stop_reason"], d["control_steps"], "/", d["reference_samples"],
      "max_err_mm=", d["max_error_mm"], "final_err_mm=", d["final_error_mm"])
```

`path_complete` + low `final_error_mm` = clean run. Anything else, go to
`predicted_beam_positions.jsonl` and check `qp_status` was `solved` at
every tick up to the stop (see section 3/6) before assuming it's a
controller/model problem rather than an externally-tuned safety box.

### 7.2 Comparing two or more runs: clearance + flange excursion + tracking

This is the pattern used to compare every rep in the 2026-09-30 study.
`magnet_xyz_m` and `q_cmd_k` are logged per tick, so you can reconstruct
both the magnet-to-exclusion-point clearance and the flange (redundant-DOF)
excursion without touching the robot:

```python
import json, numpy as np
from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik

C = np.array(json.load(open("<run_dir>/controller_metadata.json"))["magnet_exclusion_point_R"])
dh = urik.corrected_dh_from_config(urik.CONFIG)

recs = [json.loads(l) for l in open("<run_dir>/predicted_beam_positions.jsonl")]
clearances, flange_xyz = [], []
for r in recs:
    clearances.append(np.linalg.norm(np.array(r["magnet_xyz_m"]) - C))
    fk = urik.forward_kinematics(np.array(r["q_cmd_k"][:6]), dh)
    flange_xyz.append(fk.T_R_F[:3, 3])
clearances, flange_xyz = np.array(clearances), np.array(flange_xyz)
print("min/mean clearance (mm):", clearances.min()*1e3, clearances.mean()*1e3)
print("flange y range:", flange_xyz[:,1].min(), flange_xyz[:,1].max())
```

Run this over every rep dir and compare -- a bimodal split in flange y
range across reps of the SAME condition (e.g. one basin clusters near
-0.56, another near -0.42) is a real finding about the controller's
redundant-DOF resolution, not noise; see section 7.4/7.5 for how to explain
it, not just observe it.

### 7.3 Reconstructing the TCP-workspace margin (why did `tcp_out_of_workspace` trip, and on which face?)

The generic TCP box (section 0 layer 4) is **not** in the QP, so there is
no dual variable or horizon prediction for it -- reconstruct it directly
from FK against the box bounds in this script's `main()`
(`cfg.workspace_xyz_min_m`/`max_m`):

```python
import numpy as np
WS_MIN = np.array([0.4266, -0.7854, 0.3067])   # read the actual values from main()
WS_MAX = np.array([0.7115, -0.4201, 0.4043])
FACES = ["x_lo","y_lo","z_lo","x_hi","y_hi","z_hi"]

def h_ws(tcp_xyz):
    m = np.concatenate([tcp_xyz - WS_MIN, WS_MAX - tcp_xyz])
    return m.min(), FACES[m.argmin()]
```

Apply this to every tick's flange FK (as in 7.2) and you get a signed
margin + which face is binding. In the 2026-09-30 study this immediately
showed all 3 no-contact failures crossing zero on the exact same face
(`y_hi`) the trip message reported, with a **persistent monotonic decline**
over the final ~13-16 ticks (not a sudden spike) -- while every contact run
stayed >=7mm clear on a completely different face the whole time. That
temporal signature (long oscillatory phase, then a clean monotonic run to
zero) is worth checking for in your own data before concluding a trip was
"sudden."

### 7.4 The gold-standard mechanism comparison: evaluate BOTH Jacobians at the SAME state

Comparing a contact run against a no-contact run directly confounds the
model difference with the two runs' different trajectories. The fix:
whatever run you're analyzing, build **both** Jacobian providers and
evaluate both at every tick's actually-visited state, regardless of which
one the live controller used:

```python
from proper_research.hardware.online.vessel_stage_a.analyze_contact_mechanism import build_adapters
# before importing, override its module-level LUMEN_FILE/INSERTION_MAX_M/Z_RAISE_M
# constants for YOUR plan -- they default to an older plan's values.
adapter_contact, adapter_nocontact = build_adapters()

import numpy as np
from proper_research.controllers.beam_jacobian_providers import position_jacobian_from_output
z = np.array(rec["z_meas"])  # [q(6), insertion(1)] for one tick
J_contact    = position_jacobian_from_output(adapter_contact.continuous_output_jacobian(z))
J_nocontact  = position_jacobian_from_output(adapter_nocontact.continuous_output_jacobian(z))
```

Each `continuous_output_jacobian` call is ~0.5-1s (it's a real equilibrium
solve + analytic derivative, not a lookup) -- budget accordingly for a
whole run (hundreds of ticks x 2 models). `analyze_contact_mechanism_v2.py`
in this directory has a ready-made `beam_contact_profile(adapter,
lumen_query, q6, L_m)` that additionally forward-solves the FULL beam
centreline (not just the tip) and returns the closest wall approach
anywhere along the beam, the arc-length fraction, and the local wall
tangent/normal -- use the **contact-aware** adapter's own solve as your
canonical "where is the beam actually touching the wall" proxy even when
analyzing a no-contact run's logged states, since only the contact model's
equilibrium respects the wall.

With both Jacobians at the same state, compare (see
`analyze_contact_mechanism_v2.py`'s `analyze_run` for the full recipe):
tip-motion prediction error `||dx_meas - J@dstate||` for each model
(free-space vs in-contact ticks separately -- the interesting result is an
*interaction*: similar in free space, contact-model clearly better once in
contact, not a uniform offset); the wall-normal/tangential decomposition of
that error using the contact profile's normal; and each Jacobian's
wall-normal mobility gain `||n^T J||` (does the no-contact model believe it
has escape authority through the wall that the contact model correctly
says it doesn't?).

### 7.5 The strongest single result: counterfactual replay (isolate the Jacobian as the cause, not just a correlate)

Steps 7.2-7.4 show the models predict differently. To show the model
difference actually *caused* the MPC to choose a different, worse command
-- not just that it predicts worse -- reconstruct the live QP directly
(bypassing the worker process) and solve it twice from the identical
logged state, once per Jacobian schedule, with literally everything else
(reference, weights, constraints, robust margin, clearance cost) held
numerically identical:

```python
from proper_research.controllers.mpc_delay_aware.exact_qn_zero_mpc import (
    ExactQNZeroTaskNullspaceDelayAwareMPC,
)
# construct with reference_position_jacobians=schedule_contact, and a second
# instance with reference_position_jacobians=schedule_nocontact -- identical
# `config`/`beam_config`/`magnet_exclusion`/`magnet_workspace`/
# `magnet_exclusion_clearance` kwargs otherwise. See
# magnet_exclusion_self_test.py in proper_research/controllers/mpc_delay_aware/
# for the exact construction pattern to copy, and this script's
# spawn_and_warm_worker() for where every kwarg value comes from.
```

Two things to get right, both learned the hard way in the 2026-09-30 study:

1. **`solve_delay_aware` is stateful** (it carries an EMA-filtered
   disturbance-residual estimate across calls) -- a single one-shot solve
   on a freshly constructed instance will NOT match what the live run did
   unless you replay every prior tick's real logged state through it first,
   in order, so the filter reaches the same value it had live. Replay from
   tick 0 through the tick you care about, on both counterfactual
   instances, feeding each tick's *actual logged* `z_meas`/`q_cmd`/
   `q_cmd_prev`/`insertion_m`/`measured_beam_position`/`previous_input`/
   `ref_index` (not either counterfactual controller's own solved output)
   -- this keeps both instances seeing an identical state trajectory, so
   the schedule swap is the only difference between them. It's cheap
   (~5ms/solve), so replaying a whole ~500-tick run twice takes seconds,
   not minutes.
2. Test your very first offline self-test at a state that actually
   reproduces the failure mode you're investigating -- solving from the
   pristine offline-nominal state at some reference index shows nothing if
   the real issue only appears after cumulative closed-loop drift. Reuse
   the actual logged divergent state from a real failing tick, not a
   freshly-initialized one.

Then compare the two solved commands directly -- in the 2026-09-30 study,
projecting each onto the TCP-workspace margin's gradient
(`grad(h_ws)^T u*`, section 7.3) showed the no-contact counterfactual
command was margin-decreasing on every one of the final ticks before all 3
real failures, while the contact counterfactual command was consistently
less damaging and often margin-*increasing* at the identical states
(paired comparison across all three failing runs' final ticks, Wilcoxon
p≈1.6e-6). That's the difference between "the models disagree" and "the
model choice is what caused the failure."

### 7.6 Don't test independent fix mechanisms only combined

If you're testing more than one robustness mechanism at once (e.g. the
robust margin AND the clearance cost from section 4), run all of Baseline /
A-only / B-only / A+B as **separate** live trials, not just the combined
condition -- otherwise you cannot attribute which mechanism did what, or
whether they interact (in the 2026-09-30 study, the combined condition
produced a real emergent side effect -- larger redundant-DOF excursion --
that neither mechanism alone produced enough of to trip the unrelated TCP
box).
