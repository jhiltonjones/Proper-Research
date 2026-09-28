# How to plan and run a vessel path from scratch

This covers the full workflow for the real-vessel (contact-aware) planning
campaign: digitizing a vessel with `detect_blue.py`, capturing a live start
position, trimming the centreline if needed, building the offline plan, and
running the open-loop sanity check on hardware. All scripts referenced here
live in `proper_research/hardware/online/vessel_stage_a/`.

Everything below assumes `cd /home/jack/Proper-Research`, a live robot +
camera connection, and the workspace physically raised +30mm (the
`zraise_patch.apply(30.0)` convention every script here already applies
internally — you don't need to do anything extra for that).

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
automatically (no action needed, just slower).

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

## 6. (Optional) Validate the model against real hardware

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
| `capture_live_start_position.py` | Record the robot's current pose as a plan start position |
| `trim_vessel_lumen.py` | Trim a fixed length off one end of a lumen file |
| `build_vessel_plan.py` | Run the full offline planner (Layer 1→2→3) |
| `run_open_loop_vessel.py` | Live open-loop feedforward sanity check |
| `checkpoint_beam_shape_campaign.py` | Static-checkpoint predicted-vs-measured beam-shape data collection |
| `plot_checkpoint_beam_shape_campaign.py` | Turn checkpoint data into comparison figures |
| `stop_and_perturb.py` | Earlier stop-and-perturb tip-only model validation campaign |
| `run_mpc_delay_aware_vessel.py` | Closed-loop MPC controller |
| `common.py` | Shared preflight/health-check/safe-reset helpers |
