# How to plan a shape or vessel path offline, simulate it, and run it online

The offline-planning half of this doc (sections 1-2) covers both shapes and
the vessel; the live/online half (sections 3-5) covers the generic
shape-only comparison tools this doc has always documented -- for the
vessel, or for the fuller package-specific harnesses, follow section 3's
pointers instead. See `proper_research/hardware/online/vessel_stage_a/
HOWTO_VESSEL_PLANNING.md` for the vessel's own step-by-step equivalent of
this whole doc (digitizing the lumen, building the plan, then running
online), which this doc's section 2 feeds into.

All commands assume repo root (`/home/jack/Proper-Research`) and use the venv interpreter
directly: `/home/jack/Proper-Research/.venv/bin/python3`. The bimaterial beam model
(0.2mm nitinol wire + 40mm PDMS/iron-particle composite tip, 5mm overlap,
`E_composite=1.60MPa`, `E_wire=75GPa`) is the hardcoded default in `planning_context.py` and
`model_factory.py` as of 2026-09-15 — no extra flags are needed to get it; every command
below already uses it.

## 0. Before any live (online) run — always

Robot and camera health checks live in `proper_research.hardware.online.
stage_a_common` (shared by every shape's and the vessel's run scripts), and
each package's own `common.py` wraps the camera check with that rig's own
beam-base height. Run them directly from either package -- the rectangle
one here works for any unraised-rig shape plan:
```bash
.venv/bin/python3 -c "from proper_research.hardware.online.stage_a_common import check_robot_safe; check_robot_safe()"
.venv/bin/python3 -c "from proper_research.hardware.online.rectangle_stage_a.common import check_camera_healthy; check_camera_healthy()"
```
For the vessel's raised rig, use `vessel_stage_a.common.check_camera_healthy`
instead of `rectangle_stage_a.common`'s. Check: `protective_stopped: False`,
`safety_mode: 1`, magnet-to-base distance comfortably above the 211mm floor,
and the camera check reports valid tip estimates on most/all frames (it
raises `RuntimeError` instead of printing if it doesn't).

Every hardware run script in `rectangle_stage_a`/`vessel_stage_a` also calls
its package's `common.preflight()` itself before starting (health checks +
reset to the plan's own start pose) -- don't pass `--skip-preflight` unless
you already just ran one against the SAME plan and know the robot hasn't
moved since. Offline planning (`plan_shape_path.py` / `plan_vessel_path.py`)
never touches the robot — no check needed before those.

## 1. Offline planning — shapes
Base pattern:
```bash
nohup .venv/bin/python3 -m proper_research.planning.plan_shape_path \
  --shape <shape> [shape-specific flags] \
  --point-spacing-mm 0.5 --corner-smoothing-mm 2.0 \
  --position-tolerance-mm 1.5 \
  --tangent-error-selection-weight 0.0 \
  --magnet-exclusion-reference-joints="-0.8294642607318323,-1.9672757587828578,-1.730027437210083,-1.04366888225589,1.5631502866744995,-2.029626194630758" \
  --maximum-global-nodes 400 \
  --output-root <output-dir> \
  > <log-file>.out 2>&1 &
disown
```
`--shape` choices: `square`, `rectangle`, `triangle`, `line`, `circle`, `s_curve`. Shape-specific flags:
| Shape | Extra flags | Notes |
|---|---|---|
| Triangle | `--size-mm <depth> --triangle-base-mm <base> --triangle-apex-at-start --closed` | e.g. 20mm base: `--size-mm 14 --triangle-base-mm 20` |
| Rectangle (closed) | `--size-mm <width> --rect-depth-mm <depth> --closed` | e.g. `--size-mm 10 --rect-depth-mm 15` |
| U-shape (open rectangle) | `--size-mm <width> --rect-depth-mm <depth>` | same as rectangle but **omit `--closed`** |
| Circle | `--size-mm <diameter> --closed` | |
| S-curve | `--size-mm <width> --s-curve-length-mm <length>` | |

Takes 5-20 minutes depending on shape complexity (triangle/rectangle with sharp corners take
longer than circle/S-curve). Monitor with:
```bash
tail -f <log-file>.out
```
Success looks like a `[done] .../time_parameterized_configuration_path.npz` line with sample
count/duration, plus a `[done] run online with: ...` block naming the `plan_dir` to use below.
A "SAVED INVERSE AUDIT" line with `worst_tangent_node ... error=17X deg` at a sharp corner is
normal (see `beam-lateral-authority-limit` memory / this session's offline-tangent-error
investigation) — it does NOT block feasibility, only `feasible=True/False` and Layer 2's
`constraint_violation` matter for whether the plan is usable.

## 2. Offline planning — a real digitized vessel
First produce `vessel_lumen_robot_frame.json` via `detect_blue.py` (a separate,
vision-side step, not covered here). Then:
```bash
nohup .venv/bin/python3 -m proper_research.planning.plan_vessel_path \
  --lumen-file vessel_lumen_robot_frame.json \
  --insertion-max-mm 65 \
  --position-tolerance-mm 3 \
  --output-root <output-dir> \
  > <log-file>.out 2>&1 &
disown
```
`--position-tolerance-mm 3` (looser than shapes' 1.5mm) is the value used for the real-vessel
campaign started 2026-09-15 — a real lumen is noisier than a synthetic shape target.
**Known issue (2026-09-15, unresolved):** the shared starting-insertion constant `L_cmd` in
`proper_research/simulation/simulations/initial_conditions.py::make_initial_poses()` is
currently `0.025` (25mm). Do not change it to values below ~25mm without testing first — 20mm
was confirmed to fail the Layer 1 chain-rule Jacobian validation (93.8% relative error vs. the
9% ceiling) while 25mm passes cleanly on the identical vessel config. Root cause not found.

## 2b. Simulate before hardware (optional, shapes only today)

`proper_research/simulation/simulations/simulate_time_parameterized_configuration_mpc.py`
runs a closed-loop MPC simulation (no robot/camera) against a plan. It's the
cheapest way to sanity-check a new plan or a gain change before spending
robot time:
```bash
python -u -m proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc \
    --horizon 15
```
**Known gap:** its `--reference` flag is defined but never read -- it always
simulates against `build_planning_context()`'s own current plan (a shapes
planning context), not an arbitrary `--plan-dir` you pass. There is no
vessel-context equivalent anywhere under `proper_research/simulation/` today
-- simulate-before-hardware currently only works for shapes, not the vessel.
`--self-test` runs a deterministic controller-only check with no plan needed
at all, useful just to confirm the MPC machinery itself still imports and
solves.

## 3. Live open-loop test

Two separate ways to run things live exist in this codebase -- know which
one you're using:

- **This doc's `open_loop_playback.py` / `compare_controllers_live.py`
  (below)**: generic, shape-only, quick multi-controller comparison tools.
  They assume an unraised rig (`proper_research.rig_calibration.BEAM_BASE_XYZ_M`
  directly) and have none of the vessel's magnet-exclusion/raised-pivot
  machinery -- **do not use these for a vessel plan.**
- **`rectangle_stage_a`'s and `vessel_stage_a`'s own dedicated run scripts**
  (`run_open_loop_c.py`/`run_open_loop_vessel.py`,
  `run_inv_2dof_trim.py`/`run_inverse_jacobian_online_vessel.py`,
  `run_mpc_ltv.py`/`run_mpc_delay_aware_vessel.py`, and more) -- the fuller,
  validated, package-specific harnesses with each rig's own safety
  machinery and execution-layer-C tuning baked in. See
  `proper_research/hardware/online/rectangle_stage_a/README.md` and
  `proper_research/hardware/online/vessel_stage_a/HOWTO_CLOSED_LOOP_MPC.md`
  / `HOWTO_INVERSE_JACOBIAN.md` / `HOWTO_VESSEL_PLANNING.md` for those --
  **this is almost always what you want**, and the only option for a vessel
  plan.
```bash
PYTHONPATH=/home/jack/Proper-Research nohup .venv/bin/python3 -m proper_research.hardware.online.open_loop_playback \
  --plan-dir <output-dir>/<shape_or_vessel_subdir>/time_parameterized_configuration_path \
  --speed-scale 1.0 --reps 1 \
  --out-dir close_loop_logs/<run-name> \
  > <log-file>.out 2>&1 &
disown
```
`--reps N` to run multiple back-to-back (aggregated in `<out-dir>/summary.json`). Watch for
`[open-loop] DONE -> ...` and check `summary.json`'s `rms_track_mm` / `max_track_mm` /
`final_mm`.

## 4. Live closed-loop controller comparison (inv-Jacobian vs. MPC-LTI vs. MPC-LTV)
```bash
nohup .venv/bin/python3 -m proper_research.hardware.online.compare_controllers_live \
  --controllers inv,lti,ltv --reps 3 --horizon 15 --beam-len-mm 25 \
  --feedforward --force-mpc-feedforward --mpc-state-tracking-weight 0.001 \
  --plan-dir <output-dir>/<shape_or_vessel_subdir>/time_parameterized_configuration_path \
  --out-dir close_loop_logs/<run-name> \
  > <log-file>.out 2>&1 &
disown
```
**These three flags are load-bearing, not optional defaults** — validated across this whole
session (day2 20mm-triangle root-cause + this session's horizon investigation):
- `--horizon 15` — 30 causes MPC hold-phase blowup (confirmed on the 8mm triangle); untested
  below ~10.
- `--mpc-state-tracking-weight 0.001` — the default (1.0, ≈13,000/rad² effective) makes MPC+FF
  worse than open loop; 0.001 is the fix (exact 0 breaks DARE terminal-cost solvability).
- `--force-mpc-feedforward` — MPC auto-disables FF by default (it tracks its own planned
  state); this flag re-enables it for a fair comparison against inv+FF.

Do **not** add `--ff-trim-base-switch-at-hold` — tested 2026-09-15 on the U-shape, causes
6-9mm hold/final errors (all controllers), a confirmed regression, not a fix for anything.

`--beam-len-mm 25` should match whatever insertion length the plan actually starts at (see
`L_cmd` above — currently 25mm for every shape/vessel plan this session). `--reps N` sets reps
per controller; `--controllers` accepts any comma-separated subset of `inv,lti,ltv`. Progress
prints `rep K/N   <controller>` headers and a `[metrics] <controller> repK: rms=... max=...
hold=... final=... mm` line per completed rep; a final aggregate table plus `summary.json` /
`results.csv` / `compare_bars.png` / `compare_paths.png` are written to `--out-dir` at the end.

**Known transient fault:** an `RTDEControlInterface ... Robot is disconnected` error
occasionally happens mid-run (RTDE reconnect fragility, not a real fault) — the script retries
automatically (`rc=4`, 30s settle, retry) and has always recovered so far this session. If a
run instead shows a catastrophically large error (order 1cm+) immediately after a reconnect,
stop and run the `check_robot_safe()` check from section 0 before trusting the result or
continuing.

## 5. Reading results
- `close_loop_logs/<run-name>/summary.json` — mean±std per controller (`rms_track_mm`,
  `max_track_mm`, `mean_hold_mm`, `final_mm`), computed over the transit phase only
  (`terminal_hold == False` for RMS/max; hold/final use the terminal-hold ticks / last tick).
- `close_loop_logs/<run-name>/runs/<controller>_rep<N>_<timestamp>/path_follow.jsonl` — full
  per-tick diagnostics (joint corrections, solver status, Jacobian condition number, etc.) for
  deep debugging.
- `close_loop_logs/<run-name>/runs/<controller>_rep<N>_<timestamp>/tip_trajectory.csv` — raw
  tip position/error/joint trace, one row per control tick.
