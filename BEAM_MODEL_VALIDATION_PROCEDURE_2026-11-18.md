# Bounded Static Beam-Model Validation Procedure

**Status: procedure document for later supervised execution. Not executed as part of producing this document — no hardware motion occurred in the writing of this file.**

**Date**: 2026-11-18
**Plan**: `plans/rectangle_10x15mm_skipglobal_z4b_marginfix_2026-11-11/time_parameterized_configuration_path`
(230 samples total, `state` array shape (230,7) = 6 UR10e joints + 1 insertion length)

## 1. Selected sample indices

Five indices, spanning path progress and the plan's own insertion-length range
(25.0-41.3mm across all 230 samples) — a manageable subset for manual supervised
measurement, not all 230 samples.

| idx | t_s | planned q (rad) | planned L (mm) | why selected |
|---|---|---|---|---|
| 0 | 0.00 | [-0.8533, -1.8744, -1.7319, -1.1345, 1.5626, -2.0535] | 25.000 | path start, min insertion in this subset |
| 57 | 5.70 | [-0.8507, -1.8677, -1.7328, -1.1515, 1.5354, -2.0534] | 33.080 | ~25% path progress |
| 114 | 11.40 | [-0.8255, -1.8566, -1.7448, -1.1664, 1.5556, -2.0574] | 40.239 | ~50% path progress, near-max insertion |
| 171 | 17.10 | [-0.7657, -1.8104, -1.7920, -1.2319, 1.5856, -2.0692] | 36.003 | ~75% path progress |
| 229 | 22.90 | [-0.6319, -1.8001, -1.8014, -1.2518, 1.6175, -2.0781] | 27.728 | path end |

All 5 indices are `beam_sample_feasible=True` in the plan's own npz. **Coverage,
verified arithmetically against `state[:,6]`, not just restated**: the plan's true
full insertion-length range is 25.000-41.258mm; these 5 selected samples cover
25.000-40.239mm — close to the full range but **missing the top ~1.0mm** (the
plan's single longest-insertion sample near the end of its range is not among the 5
selected). This is a minor, acknowledged gap, not a claim of exact full-range
coverage.

## 2. Positioning procedure

**Checked for an existing supervised/teach-pendant positioning procedure, distinct
from the scripted calls below: none found.** Searched this codebase and prior
session docs (`SERVO_DIAGNOSTIC_LOGGING_V2_CORRECTIONS_2026-11-17.md`,
`vessel_stage_a/HOWTO_CLOSED_LOOP_MPC.md`, `SERVOJ_COMMAND_PATH_TRACE_CORRECTIONS_2026-11-17.md`,
`SERVO_DIAGNOSTIC_LOGGING_INTEGRATION_PROPOSAL_2026-11-17.md`) for pendant-specific
procedures. The only pendant references found are for switching the robot into
**Remote Control mode** before a script can command it, and for **protective-stop/
E-stop recovery** — both in `vessel_stage_a/HOWTO_CLOSED_LOOP_MPC.md`. There is no
documented manual/teach-pendant jogging procedure for positioning the robot to a
specific planned joint configuration anywhere in this codebase. Stating this plainly
rather than inventing one: **the scripted calls below are the only existing
positioning tooling found.** If a genuinely human-in-the-loop pendant procedure is
required instead, it does not exist yet and would need to be written and
documented separately before this validation procedure could rely on it.

- **Joints**: `proper_research/hardware/online/rectangle_stage_a/common.py`'s
  `reset_to_plan_initial(q0, robot_ip=...)` — already used inside `preflight()` to move
  the robot to a planned joint vector. Call it directly with each selected index's
  `state[idx, :6]`.
- **Insertion**: `proper_research/hardware/online/advancer_excitation/reset_insertion.py`'s
  `reset_insertion(target_mm, live=True)` — the existing closed-loop advancer tool,
  now running through the corrected camera chain (this session's
  `apply_affine_insertion_correction()` and the `state_stream.py` frame fix). Call with
  each index's `state[idx, 6] * 1000`.

## 3. Measurement fields, per sample

1. **Measured joint angles** — robot's own `getActualQ()` at rest, after
   `reset_to_plan_initial` settles.
2. **Measured physical insertion length** — via `measure_l0.measure_l_mm()` (the
   corrected chain), not the commanded target passed to `reset_insertion`.
3. **Camera-measured tip position** (`NewFrameTipMapper`'s `tip_position_m`) — the raw
   3D measurement.
4. 2-3 repeated readings per sample (see Section 6, noise characterization).

## 4. Offline comparison — implemented and tested this round

`proper_research/hardware/online/rectangle_stage_a/beam_model_validation_compare.py`
(new this round) takes measured `(q, L)` samples plus the fixed registration as JSON
input and outputs the three separated error numbers defined in Section 5. It is
wired to call this project's existing model-evaluation entry point,
`beam_jacobian_providers.from_model_bundle(...)` (already used project-wide for
predicting tip/beam position from a joint+insertion state, e.g. in
`vessel_stage_a/verify_contact_jacobian_implementation.py`), via a `model_predict_tip_fn`
hook — the real call requires the live plan's model bundle context, which an offline
self-test cannot build standalone, so the script's `--model mock` path uses a
deliberately crude placeholder forward model in its place for testing the plumbing.
**At execution time, the `model="mock"` branch in `run_comparison()` must be replaced
with the real `from_model_bundle(...)` call** — flagged explicitly in the script's own
`NotImplementedError` for any non-mock `--model` value, so it cannot be silently run
against the wrong model.

**Frames used, stated explicitly** (per the script's own module docstring):
- `measured_q`/`planned_q`: robot joint space (UR10e joint angles, rad).
- `measured_L`/`planned_L`: insertion length (m), advancer-axis scalar.
- `measured_tip_R`: robot base frame R (m), from the live camera chain.
- Model-predicted tip: computed in planner frame P, then rotated into frame R via
  the **fixed** `(r_fit, t_fit)` registration (read from a JSON file passed in, never
  refit inside this script) before comparison — both tip quantities compared are in
  frame R.

**Offline self-test, run this round** (synthetic measured samples at the 5 selected
indices, small Gaussian joint/insertion noise injected around the plan's own planned
values, registration taken from a real analyzed run's `controller_metadata.json`):
the script ran end-to-end without error and produced three distinct, separated
numbers per sample, e.g. for idx=0: joint positioning error 0.00187 rad, insertion
positioning error 0.00039 m, model planar-tip error 0.855 m (this last number is
large and not physically meaningful here — the mock forward model is a deliberate
placeholder, not the real beam model; the self-test confirms the pipeline and error
separation work correctly, not beam-model accuracy). This confirms the CLI is
functional and ready for the `--model` real-model wiring at execution time, not
left unwritten.

## 5. Critical methodological requirements

These are binding requirements for how this procedure must be executed, not optional
reminders:

- **Evaluate at MEASURED q and MEASURED L, never planned/commanded values.** The model
  prediction must use the robot's actual post-settle joint readout and the actual
  post-`reset_insertion` camera-measured length, not `state[idx, :6]`/`state[idx, 6]`
  directly. This tests the model against reality, not against itself.
- **Fixed registration.** Use the one `planner_to_live_t_fit` / frame registration
  already established for whichever run context this procedure is executed under, held
  constant across all 5 samples. Do not refit the planner-to-robot registration per
  sample.
- **THREE separate numbers per sample, never conflated into one "total error":**
  - *(a) Joint positioning error* = `||measured_q - planned_q||` (rad) — did the
    robot's joints get where they were told to go (a control-execution question).
  - *(b) Insertion positioning error* = `|measured_L - planned_L|` (m) — did the
    advancer get to the planned insertion length (a separate control-execution
    question, different actuator/failure mode from (a)).
  - *(c) Model planar-tip error* = `||model-predicted tip_xy(measured q, measured L) -
    camera-measured tip_xy||`, in the camera's 2D measurement plane only — does the
    model predict reality correctly, given where the robot/advancer actually are (a
    model-fidelity question). Explicitly NOT a 3D tip error — see the observable-
    dimensions note below.
- **Observable dimensions, stated honestly.** This project's camera chain gives a 3D
  `tip_position_m` via the planar-homography + pivot-based reconstruction established
  this session — but X/Y come from a genuine 2D image-plane measurement converted via
  the fixed pixel-scale calibration, while Z is computed from the fixed pivot height
  plus the detected chord geometry, not independently measured per sample. **This
  procedure can validate tip position along the camera's imaging plane well; it cannot
  independently cross-check Z against anything other than the same pivot assumption
  already baked into the frame fix.** State this limitation plainly when reporting
  results from this procedure — do not claim full independent 3D validation.

## 6. Repeated measurements (noise characterization)

Take 3 independent camera readings (fresh camera sessions, following this session's
established stability-check pattern: separate `CameraSource` start/stop per reading,
a few seconds apart) at **index 114** (chosen as the longest insertion length in this
subset, likely most sensitive to any residual geometric nonlinearity in the model).
Report the resulting measurement-noise spread (mean/std across the 3 readings)
separately from the model-error signal computed at that sample — do not let
measurement noise be silently absorbed into the model-error number.

## 7. Calibration / validation split (proposal for future use, not required now)

If beam-model parameter fitting is ever subsequently proposed using data from this
procedure, the following split is proposed now so the procedure would not need to be
redesigned later:

- **Fitting subset**: indices {0, 114, 229} — insertion lengths 25.000, 40.239,
  27.728mm (spans the full selected range, both extremes).
- **Held-out validation subset**: indices {57, 171} — insertion lengths 33.080,
  36.003mm.

**Verified arithmetically, correcting an earlier imprecise claim**: the held-out
subset's insertion-length range (33.08-36.00mm) sits **entirely inside** the fit
subset's range (25.00-40.24mm). This split tests **interpolation** within the fitted
range, not extrapolation beyond it — stated explicitly so this isn't misread as a
true generalization/extrapolation test. If extrapolation testing is wanted, the fit
subset would need to exclude one of the range extremes (e.g. fit on
{57, 114, 171} and hold out {0, 229} to test extrapolation at both ends) — proposed
here as an alternative, not adopted. This is a proposal for a possible future use of
this data, not an instruction to act on now.
