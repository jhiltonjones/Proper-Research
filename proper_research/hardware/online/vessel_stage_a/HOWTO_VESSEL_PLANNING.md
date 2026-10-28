# How to plan and run a vessel path from scratch

This covers the full workflow for the real-vessel (contact-aware) planning
campaign: digitizing a vessel with `detect_blue.py`, capturing a live start
position, trimming the centreline if needed, building the offline plan, and
running the open-loop sanity check on hardware. All scripts referenced here
live in `proper_research/hardware/online/vessel_stage_a/`.

Everything below assumes `cd /home/jack/Proper-Research` and a live robot +
camera connection.

**On the z-raise (stale note, kept for history):** this section used to
describe a `--z-raise-mm` flag shared by every script here. As of
2026-10-02 that whole indirection was removed project-wide -- every
script in this directory now uses the single, fixed, recalibrated
`build_vessel_plan.BEAM_BASE_PIVOT_Z` instead (no flag, no per-run choice
to get wrong). None of the example commands below pass `--z-raise-mm`
for this reason; if you see it in an older note or log, it refers to a
setup this doc no longer describes. The real failure mode this used to
warn about -- a script silently misreading real camera detections by a
z-offset -- is exactly what `HOWTO_CLOSED_LOOP_MPC.md` section 6's
frame-registration symptom still covers, just from a different, no-longer-
possible-to-mismatch cause.

## 0. Background: what each file represents

- **Lumen file** (`vessel_lumen_robot_frame*.json`): the digitized vessel
  centreline + radius, in **robot frame R** — this is both the contact
  model's wall geometry AND the tip-tracking target (there's no separate
  "target path"; the vessel's own centreline is the shape).
- **Start-position JSON** (`vessel_magnet_initial_position_*.json`): the
  source magnet's pose the offline plan starts from, plus the
  magnet-to-beam-base safety floor. These are recorded as **two separate
  numbers on purpose** — see the warning in step 3.
- **Plan directory** (`plans/<name>/time_parameterized_configuration_path`):
  the final output — a time-parameterized joint/insertion trajectory ready
  to run on hardware.

## 1. Digitize the vessel with `detect_blue.py`

If you don't already have a lumen file for the current physical vessel
setup, use the interactive click tool:

```python
from proper_research.vision.detect_blue import draw_vessel_lumen_for_planner
draw_vessel_lumen_for_planner(
    image_filename="focused_image.jpg",   # a fresh photo of the vessel setup
    save_path="vessel_lumen_robot_frame.json",
)
```

Click the vessel's left/right walls (or its centreline directly — press
`m`) on the displayed image; press `s` to save. This reuses the **existing**
beam-frame calibration (`manual_vessel_boundaries.json` +
`calibration_points.json`) — the same frame the live tip tracker and every
other model in this project shares — so the new vessel lands in a frame
that's already consistent with everything else. It does not re-click or
modify that calibration.

The saved file has `lumen_C_m` (N×3, metres, frame R) and `lumen_R_m` (N,)
plus a `provenance` block recording exactly which calibration files were
used.

**If the vision ROI/marker-detection coordinates need correcting** (e.g.
`custom_area.json`, the red-marker search polygon), edit that file directly
— it's independent of the lumen digitization above.

## 1b. (If the physical vessel got bumped) Re-align it against the existing lumen file

If the vessel phantom moved but you still trust the existing lumen file's
*shape* (just not its position), you don't need to re-digitize from
scratch — use the live overlay tool to physically move the vessel back
into place instead:

```bash
python3 -m proper_research.hardware.online.vessel_stage_a.live_vessel_alignment_overlay \
    --lumen-file vessel_lumen_robot_frame.json
```

This opens a live camera window with the lumen's centreline/walls
(cyan/green/red) drawn on top, projected through the **same** calibration
the live vision pipeline actually uses (`NewFrameTipMapper`'s `T_R_B` +
`PlanarPixelCalibration` — not an approximation). Move the vessel until the
drawn lines match its real walls, press `q`.

If the vessel's shape itself changed (not just position) — or you don't
trust the old digitization — just re-run step 1 instead and overwrite the
lumen file, which is exactly what happened 2026-10-01: a fresh
`vessel_lumen_robot_frame.json` was captured, which is a **different file**
from any `_zraise42`/`_raised3cm`/etc. variant already sitting in the repo
— check the file's own mtime and `provenance` block before reusing one you
didn't just capture yourself, to make sure you're building against the
vessel's *current* position, not a stale one.

## 1c. (Optional) Shift the lumen centreline toward one wall (offline configuration change)

This is the heavyweight way to bias the plan toward one wall -- it
changes what Layer 1's optimizer is actually solved against (and what
the contact-aware model treats as "the wall"), not just what the live
controller tracks. If you only want to nudge a single closed-loop run's
tracking target without rebuilding anything, use
`run_mpc_delay_aware_vessel.py --right-shift-mm` instead -- see
HOWTO_CLOSED_LOOP_MPC.md section 2b, which contrasts both options
directly.

```bash
python -m proper_research.hardware.online.vessel_stage_a.shift_lumen_centerline \
    --lumen-file vessel_lumen_robot_frame.json \
    --shift-mm 1.0 --direction right \
    --out vessel_lumen_robot_frame_right1mm.json
```

Same tangent/normal "right" convention as the live overlay tool in
section 1b (right = centreline shifted opposite the +90°-rotated local
tangent); radii are left unchanged, only the centreline moves. After
shifting, treat the output file as a brand-new lumen file: rebuild the
plan from it (step 4 below) and
rebuild both Jacobian schedules from the new plan before running closed-loop
MPC (HOWTO_CLOSED_LOOP_MPC.md section 2b, "Forcing a schedule rebuild") --
the schedule cache does not know the lumen changed and will silently keep
serving the old, now-wrong schedule if you reuse an old cache path.

## 2. (Optional) Trim the lumen if the plan shouldn't reach the true end

```bash
python -m proper_research.hardware.online.vessel_stage_a.trim_vessel_lumen \
    --lumen-file vessel_lumen_robot_frame.json \
    --trim-mm 6.0 --end end \
    --out vessel_lumen_robot_frame_trimmed6mm.json
```

`--end end` trims off the distal end (furthest from the beam base);
`--end start` trims off the entrance. The cut point is interpolated exactly
at the requested arc length, not just snapped to the nearest sample.

## 3. Capture the start position

Jog the robot (manually, or however you like) to wherever you want the
offline plan to start from, then record it:

```bash
python -m proper_research.hardware.online.vessel_stage_a.capture_live_start_position \
    --out vessel_magnet_initial_position_live.json \
    --exclusion-floor-mm 210.43
```

**Read this before choosing `--exclusion-floor-mm`.** The exclusion floor
is the hard safety constraint "the source magnet may never come closer to
the beam base than this distance" — it feeds
`magnet_beam_base_exclusion_radius_m` in the planner. It is a **separate
number from the start position's own distance to the beam base**, and this
matters:

- If you omit `--exclusion-floor-mm`, this script sets the floor equal to
  the captured position's own distance — i.e. the start position sits
  *exactly on* the constraint boundary, with **zero slack**. We hit this
  exact bug on 2026-09-27: the planner immediately pins the magnet to the
  floor at every node (confirmed via an instrumented diagnostic — mean
  margin 0.03mm, repeatedly touching 0.000mm), leaving no room to
  manoeuvre and causing solves to hang or fail deep into the path.
- The fix that worked: capture a start position with **real margin** above
  whatever floor you're using (e.g. retreat 5-15mm further from the beam
  base than the floor requires), or pass an explicit
  `--exclusion-floor-mm` that's comfortably below the captured position's
  own distance.
- Do **not** lower the floor below a value you haven't personally verified
  safe on the real hardware (no protective stop, clear of the table) — this
  number is a physical safety limit, not a tuning knob.

## 4. Build the offline plan

```bash
python -m proper_research.hardware.online.vessel_stage_a.build_vessel_plan \
    --lumen-file vessel_lumen_robot_frame_trimmed6mm.json \
    --start-position-json vessel_magnet_initial_position_live.json \
    --insertion-max-mm 65 \
    --output-root plans/my_vessel_plan
```

This is slow (Layer 1 does one real nonlinear beam solve per centreline
node) — run it in the background (`nohup timeout 7200 python3 ... &`) and
watch the log. `all_nodes_feasible=True` at the end of Layer 1 means it's
done; `False` means it fell through to Layer 2's recovery optimizer
automatically (no action needed, just slower). Budget real time for this:
a 726-sample/80mm-insertion-max plan (2026-10-01) took **~1h48m** for Layer
1 alone (6490s, logged as `[layer1] done in Ns`) before Layer 3's much
faster time-parameterization pass — don't assume the ~90-150s figure
quoted for *schedule* builds elsewhere in this doc set applies to the
offline *plan* build; they're different stages with very different costs.

Two things baked into this script that you generally shouldn't need to
touch, but are worth knowing about:

- **`--maximum-function-evaluations 60`** (default): the library default
  (300) let a single hard node hang for 60+ minutes, 3 times in a row, on
  2026-09-27 (traced to `scipy.optimize`'s `maxiter`, which really does
  bound the *outer* solve — the hang was CPU contention from other
  simultaneous background jobs slowing convergence, not an actual infinite
  loop). A smaller budget makes any one node fail fast instead, and
  Layer 2's `recover_partial` repairs whatever Layer 1 couldn't reach.
- **`--finite-difference-joint-step-rad 1.5e-2` / `--finite-difference-insertion-step-m 5e-3`**:
  the library default step (1e-6) is far smaller than the contact-aware
  beam solver's own convergence precision (`optimizer_gtol=1e-5`), so the
  chain-rule sanity check at 1e-6 measures pure solver noise, not the
  analytic Jacobian's real accuracy — we measured ~100% "error" at 1e-6 vs.
  a genuine ~9% plateau at this step size (matches the ~5-10%-off-FD figure
  already documented elsewhere in this codebase for the same effect). This
  is **not** a bug in the analytic Jacobian; it's a validation-step-size fix.

**Also check `all_nodes_feasible` against the magnet exclusion floor if it
fails early** — the vessel-CENTRELINE magnet exclusion
(`source_magnet_lumen_exclusion_radius_m`) is deliberately left off
(`None`) in this script; only the fixed beam-base floor is enforced. If you
need the centreline exclusion too, you'll need to edit the script (it's a
different, stricter constraint most vessel plans don't need).

## 5. Sanity-check with the open-loop run (do this before any closed loop)

```bash
python -m proper_research.hardware.online.vessel_stage_a.run_open_loop_vessel \
    --plan-dir plans/my_vessel_plan/time_parameterized_configuration_path \
    --out-dir close_loop_logs/myrun \
    --run-name my_vessel_plan_openloop \
    --exclusion-floor-mm 210.43
```

This streams the plan's own feedforward commands with no correction — it's
the acceptance test that the execution layer faithfully reproduces the
planned motion, run before trusting any closed-loop controller through the
same layer. It automatically resets the robot to the plan's own start state
(joints from the plan's `state_reference[0]`, insertion from `L0`) and
checks the reset motion doesn't violate the exclusion floor mid-transit,
retrying through a retreat waypoint if needed.

**Before this step**, reset the physical insertion length if it's drifted
from a previous run — the advancer has no encoder, so nothing moves unless
explicitly commanded:

```python
from proper_research.hardware.online.vessel_stage_a.checkpoint_beam_shape_campaign import (
    reset_insertion_to_target,
)
from pathlib import Path
reset_insertion_to_target(25.65, live=True, out_dir=Path("/tmp"))  # target L0 in mm
```

(`check_camera_healthy`, called inside `run_open_loop_vessel.py`'s
preflight, will refuse to proceed anyway if the measured and expected
insertion differ by more than `--insertion-tol-mm`, so this catches drift
even if you forget — but retracting first avoids the failure.)

**If the robot is currently far from the NEW plan's start pose** (e.g. you
just finished a different plan, or the robot was jogged manually), the
straight-line reset path preflight checks can refuse outright with
`RuntimeError: refusing reset: ... would violate a magnet safety
constraint`. This is routine, not a sign anything is wrong — the preflight
would rather refuse than risk cutting through the exclusion zone
mid-motion. `run_open_loop_vessel.py` already retries through a retreat
waypoint automatically; `run_mpc_delay_aware_vessel.py` does **not** (see
`HOWTO_CLOSED_LOOP_MPC.md` section 6 for the manual fix).

**Reading the result**: don't just check `stop_reason`, look at the
tracking-error plot/CSV. A plan that completes open-loop with a small,
genuinely random-looking error is healthy; a plan whose error grows
**steadily in one direction** (e.g. one axis alone drifting several mm
while the others stay flat) over the course of the path is worth plotting
properly before trusting it — top-down (x,y) path vs. desired, and each
error component vs. insertion depth, are usually enough to tell a real
frame/registration offset (near-constant or path-shape-mismatched error
from the start) apart from ordinary open-loop feedforward model error
(error that tracks the path shape well but accumulates gradually with
insertion depth — exactly what closed-loop MPC exists to correct). A
~3-4mm final error on an ~70-80mm-insertion open-loop run is not unusual
for this beam and is not on its own a reason to suspect a calibration bug.

## 6. Run the closed-loop MPC

Once the open-loop check passes, see `HOWTO_CLOSED_LOOP_MPC.md` in this
same directory for the full closed-loop controller workflow
(`run_mpc_delay_aware_vessel.py`) -- prerequisites, CLI flags, safety
monitors (including the magnet-exclusion constraint now wired directly
into the QP), and how to read the run's output.

## 6b. (Optional) Run the closed-loop inverse-Jacobian controller instead

The naive-controller comparator against the MPC run above -- see
`HOWTO_INVERSE_JACOBIAN.md` in this same directory for prerequisites (it
reuses the MPC run's own `--schedule-cache`, it does not build one), CLI
flags, and the three magnet-protection modes (`clip`/`hold`/`selective`).

## 7. (Optional) Validate the model against real hardware

`checkpoint_beam_shape_campaign.py` and
`plot_checkpoint_beam_shape_campaign.py` run a static-checkpoint
predicted-vs-measured full-beam-shape comparison (not just tip), including
a contact-vs-no-contact model comparison, across a spread of insertion
lengths. See that script's own docstring for usage — it's a good next step
if you change the vessel geometry, magnet position, or plan significantly
and want to check the model still tracks reality before trusting it.

## Summary of scripts in this directory

| Script | Purpose |
|---|---|
| `live_vessel_alignment_overlay.py` | Live camera overlay of an existing lumen file, for physically re-aligning a moved vessel |
| `capture_live_start_position.py` | Record the robot's current pose as a plan start position |
| `trim_vessel_lumen.py` | Trim a fixed length off one end of a lumen file |
| `shift_lumen_centerline.py` | Rigidly shift a lumen file's centreline toward one wall (offline configuration change; see section 1c) |
| `build_vessel_plan.py` | Run the full offline planner (Layer 1→2→3) |
| `run_open_loop_vessel.py` | Live open-loop feedforward sanity check |
| `checkpoint_beam_shape_campaign.py` | Static-checkpoint predicted-vs-measured beam-shape data collection |
| `run_mpc_delay_aware_vessel.py` | Closed-loop MPC controller |
| `run_inverse_jacobian_online_vessel.py` | Closed-loop inverse-Jacobian controller (current -- contact model + magnet-protection modes), see `HOWTO_INVERSE_JACOBIAN.md` |
| `run_inverse_jacobian_vessel.py` | Closed-loop inverse-Jacobian controller (older, no-contact only), see `HOWTO_INVERSE_JACOBIAN.md` |

One-off diagnostic/plotting scripts (`analyze_*`, `measure_jcam_*`,
`validate_*`, `plot_checkpoint_beam_shape_campaign.py`,
`stop_and_perturb.py`, and others) have been moved into `archive/` -- each
answers a specific dated question, see its own docstring; none of them are
part of the regular plan-then-run workflow above.
| `common.py` | Shared preflight/health-check/safe-reset helpers |
