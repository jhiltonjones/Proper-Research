# HOWTO: Campaign 2 (single entry point)

Written 2026-11-11, third revision same day. Source of truth for settings:
`protocol_campaign2.yaml` at the repo root (DRAFT, **not yet reviewed or
signed off by the author** -- the author reviews and commits this
themselves; nothing here signs it off).

## Campaign order

1. Finish code gates + make instructions reproducible (this document).
2. K2 dry run on the real arm, no MCR (Section E).
3. Day-1 pilots + go/no-go gate (Section F).
4. Core campaign: vessel plan B, rectangle, triangle-35 -- 15 conditions (Section C).
5. Trap paths -- **planned, later phase, not dropped** (Section G).
6. Silicone demonstration -- **planned, final phase**, author's own scoping (Section G).

## Closed decision: shape rig keep-out/height-band

**Author decision (2026-11-11), recorded, not reopened, author's exact
words:** "I accept the measured TCP box as the shape rig's safety
mechanism." The TCP box
(`rectangle_stage_a.common.WORKSPACE_XYZ_MIN_M/MAX_M` =
`(0.24469, -0.89361, 0.36544)` to `(0.58971, -0.49576, 0.46496)`) **is**
the shape rig's keep-out/height-band mechanism, by this decision -- no
separate mechanism needs implementing. The one known exception is
unchanged: the shared start pose clears the z-face by only 24.73mm
(documented in `common.py`/`BOX_ADJUSTMENT_NOTE.txt` since Round 33/AF1).
This decision is why K2 (Section E, step 2) now includes a physical
clearance check -- it is what validates the decision on real hardware.

## Shape MPC runner: `run_mpc_delay_aware_insertion_anchor.py`

Not an ablation, unlike `qn_zero.py` (explicit Q_N=0 ablation) or
`nullq_r700.py` (superseded). `insertion_anchor.py` is the first
corrected validation after the 2026-09-21 frame-registration fix, at the
real formulation (`gamma=0, R=700*R0`, Q_N **on**). Its metadata writer
sets `controller_variant: "nullspace_Q_R700_frame_fix"` -- this field (not
`controller_kind`, which is identical across all 10 scripts) is what
distinguishes which script produced a given log.

## Open gaps -- NOT closed, do not call this gate "complete"

- **INV-7 has no mocked-loop coverage** (built for MPC-SJ/FJ and vessel
  only this session).
- **INV-7's runner writes no `controller_metadata.json`** at all (a
  separately-known gap, `summary.json["config"]` is its only source).

Neither blocks K2 (which only exercises rectangle MPC + delay re-ID) or
the Day-1 pilots (which have no INV-7 condition). Both must close, or be
explicitly waived by the author in writing, before INV-7 runs in the core
campaign (Section C).

## Section A -- Offline planning (no robot)

```bash
# Rectangle: entry point plan_shape_path.py, calls build_planning_context()
# then the L1/L2/L3 pipeline (offline_inverse_configuration_head_exclusion.py
# -> run_global_configuration -> run_time_parameterization). Output:
# time_parameterized_configuration_path/{*.npz,*.csv,*.png,*_summary.json}.
python3 -m proper_research.planning.plan_shape_path \
  --shape rectangle --size-mm 10 --rect-depth-mm 15 --skip-global \
  --use-calibrated-beam-base \
  --output-root plans/rectangle_10x15mm_skipglobal_z4b_2026-11-11
# --use-calibrated-beam-base: build_planning_context() otherwise defaults
#   to the legacy stale pivot (kept only for backward compatibility).

# Triangle-35: same entry point, same code path.
python3 -m proper_research.planning.plan_shape_path \
  --shape triangle --size-mm 35 --skip-global \
  --use-calibrated-beam-base \
  --output-root plans/triangle_base35mm_skipglobal_z4b_2026-11-11
```

**To verify the pipeline without risking the real campaign plans, rebuild
into a scratch directory and hash-compare before ever promoting it:**

```bash
# Rebuild into a scratch dir, NEVER the real campaign-plan path
python3 -m proper_research.planning.plan_shape_path \
  --shape rectangle --size-mm 10 --rect-depth-mm 15 --skip-global \
  --use-calibrated-beam-base \
  --output-root plans/_scratch_rectangle_verify_2026-11-11

# Compare the rebuild's hash against the real campaign plan's hash
python3 -c "
import hashlib
a = hashlib.sha256(open('plans/_scratch_rectangle_verify_2026-11-11/time_parameterized_configuration_path/time_parameterized_configuration_path.npz','rb').read()).hexdigest()
b = hashlib.sha256(open('plans/rectangle_10x15mm_skipglobal_z4b_2026-11-11/time_parameterized_configuration_path/time_parameterized_configuration_path.npz','rb').read()).hexdigest()
print('scratch:', a); print('campaign:', b); print('MATCH' if a==b else 'DIFFER -- do NOT promote without understanding why')
"
# Only if MATCH, and only if you have a real reason to replace the campaign
# plan (e.g. a code fix that should be a no-op): cp the scratch files over
# the campaign ones by hand, never by re-running --output-root pointed at
# the campaign path directly.
rm -rf plans/_scratch_rectangle_verify_2026-11-11   # clean up the scratch dir
```

**Hash check against the protocol's recorded values:**

```bash
python3 -c "
import hashlib
for p in [
  'plans/rectangle_10x15mm_skipglobal_z4b_2026-11-11/time_parameterized_configuration_path/time_parameterized_configuration_path.npz',
  'plans/triangle_base35mm_skipglobal_z4b_2026-11-11/time_parameterized_configuration_path/time_parameterized_configuration_path.npz',
]:
    print(p, hashlib.sha256(open(p,'rb').read()).hexdigest())
"
# rectangle must print 62cf493b0dc48518985e19fb09c97021de8434b7ca2a92a3a7a4229602453613
# triangle-35 must print 929739fd7fae369202328da478f758f32fc3e7b1a093a80178136278e1adc64c
```

## Section B -- Vessel plan: reused as-is, not rebuilt

Confirmed: the Z4b change touching `vessel_stage_a/` (counter mode,
streaming, the beam-base guard, provenance) does not touch the solve
path; the numerical gate (plan-B schedule unchanged, 20-tick re-solve
identical) passed with 0.0 difference against the current code. Plan
file unchanged, hash unchanged:
`plans/vessel_phi30_L30_left1mm_newwall_tol0p5_floor240_dipolealign_recal_matchedtarget_dipolealigned_2026-10-20/time_parameterized_configuration_path/time_parameterized_configuration_path.npz`,
SHA-256 `16bf7e34cddbba47b9a9569e1f94ebeaf31a7f18ab43356ba11000c17d506fd2`.

**`--lumen-file` and `--insertion-max-mm`'s exact original build values
were NOT found this session** -- no argv was logged for this plan's
original build (pre-provenance-push era), and the plan directory's own
`vessel_lumen/provenance.json` records calibration data, not CLI args.
State this plainly rather than inventing plausible-looking values: the
author needs to supply these from their own build records if the plan
ever needs rebuilding (it does not, for campaign 2 -- see above).

## Section C -- Running the 15 core conditions

Full per-condition settings: `protocol_campaign2.yaml`.

```bash
# Rectangle MPC-SJ N=15 -- MOVES THE ARM.
# Entry point run_mpc_delay_aware_insertion_anchor.py; main() builds
# ProcessIsolatedDelayAwareAdapter and drives it through
# close_loop_path_follow.run() (the shared per-tick scheduler/accumulator/
# monitor loop). Flags verified against the script's CURRENT argparse
# (2026-11-11), not assumed from an earlier pass:
python3 -m proper_research.hardware.online.rectangle_stage_a.run_mpc_delay_aware_insertion_anchor \
  --plan-dir plans/rectangle_10x15mm_skipglobal_z4b_2026-11-11/time_parameterized_configuration_path \
  --out-dir close_loop_logs/myrun \
  --schedule-cache <scheduled-jacobian-cache.npy> \
  --horizon 15 \
  --max-tracking-error-mm 7 \
  --tcp-workspace-robust-margin-mm 15
# --plan-dir, --out-dir, --schedule-cache: REQUIRED (argparse enforces).
# --horizon: N (default 15; pass 3 or 1 for those conditions).
# --max-tracking-error-mm: default 0.0 on this script -- MUST pass 7
#   explicitly for the campaign-2 MPC threshold (15 for INV-7, below).
# Insertion-offset abort: defaults OFF as of Z4b step 2 (2026-11-11);
#   --disable-insertion-offset-abort is now a no-op (kept for backward
#   CLI compatibility). Do not pass --enable-insertion-offset-abort
#   unless you deliberately want it on (campaign-2 decision is off).
# --tcp-workspace-robust-margin-mm 15: campaign-2 decision, inward margin
#   on the in-QP TCP-box constraint (default is already 15.0 on this
#   script -- passing it explicitly just makes the choice visible in argv).
# SJ vs FJ: selected by WHICH FILE --schedule-cache points at (a genuine
#   per-sample schedule vs. a frozen/repeated one), not a flag.
# Output: close_loop_logs/myrun/mpc_delay_aware_insertion_anchor_accumC_<ts>/
#   {controller_metadata.json, path_follow.jsonl, summary.json, ...}.
# Check it worked: summary.json["stop_reason"] == "path_complete" and
#   controller_metadata.json["controller_variant"] ==
#   "nullspace_Q_R700_frame_fix".

# Rectangle INV-7 SJ -- MOVES THE ARM.
# Entry point run_inv_7dof_delay_aware.py -> InverseJacobian7DOFDelayAware
# through the same close_loop_path_follow.run() loop.
python3 -m proper_research.hardware.online.rectangle_stage_a.run_inv_7dof_delay_aware \
  --plan-dir plans/rectangle_10x15mm_skipglobal_z4b_2026-11-11/time_parameterized_configuration_path \
  --out-dir close_loop_logs/myrun \
  --schedule-cache <scheduled-jacobian-cache.npy> \
  --max-tracking-error-mm 15
# Check: summary.json["config"]["controller_kind"] == "inv_7dof_delay_aware"
#   (this script's logs DO show this cleanly, per HOWTO_INDEX.md's audit).

# Rectangle FF (open-loop/feedforward) -- MOVES THE ARM.
# Entry point run_open_loop_c.py -- plays the plan's joint trajectory back
# directly via servoJ, no feedback controller at all.
python3 -m proper_research.hardware.online.rectangle_stage_a.run_open_loop_c \
  --plan-dir plans/rectangle_10x15mm_skipglobal_z4b_2026-11-11/time_parameterized_configuration_path \
  --out-dir close_loop_logs/myrun

# Triangle-35: identical patterns, --plan-dir pointed at
# plans/triangle_base35mm_skipglobal_z4b_2026-11-11/time_parameterized_configuration_path
# (condition set: FF, INV-7 SJ, FJ N=15, SJ N=15, SJ N=1 -- no N=3).

# Vessel MPC-SJ N=15 -- MOVES THE ARM.
# Entry point run_mpc_delay_aware_vessel.py -> ExactQNZeroTaskNullspaceDelayAwareMPC
# through close_loop_path_follow.run(). Required flags per the CURRENT
# argparse (plan-dir, out-dir, schedule-cache, lumen-file, insertion-max-mm
# are all required=True):
python3 -m proper_research.hardware.online.vessel_stage_a.run_mpc_delay_aware_vessel \
  --plan-dir plans/vessel_phi30_L30_left1mm_newwall_tol0p5_floor240_dipolealign_recal_matchedtarget_dipolealigned_2026-10-20/time_parameterized_configuration_path \
  --out-dir close_loop_logs/myrun \
  --schedule-cache <scheduled-jacobian-cache.npy> \
  --lumen-file <REQUIRED -- exact value not found this session, see Section B> \
  --insertion-max-mm <REQUIRED -- exact value not found this session, see Section B> \
  --horizon 15 \
  --max-tracking-error-mm 7
# --jacobian-source has only two choices on THIS script: "schedule"
#   (default) or "hybrid_j1j3" -- there is NO "frozen" choice. FJ is
#   selected the SAME way as rectangle: by the schedule-cache FILE
#   content (a frozen/repeated schedule), not by --jacobian-source.
# N=3: --horizon 3. INV-7: use run_inverse_jacobian_online_vessel.py,
#   same plan-dir, no --horizon flag (that script is INV-7-specific).
```

## Section D -- Four-way distinction (re-verified against current code)

1. **Fully mocked (no live arm at all):** no dedicated mode found this
   session for any of these runners; the Z4b `step7_mocked_loop.py`
   harness (below) is the closest built example, not a flag on the real
   scripts.
2. **`--dry-run`:** confirmed by this session's own source audit AND by
   the script's own help text (`insertion_anchor.py --help`): "real
   camera+robot connection, real solves, but no servoJ motion." Connects
   live to read state; commands nothing. Not a hardware-free check.
3. **Motion-free, hardware-connected checks** (K2 setup confirmation,
   pre-flight tests): arm powered and reachable, nothing commanded.
4. **Commands that move the arm:** every Section C command, every K2
   extreme-pose check (Section E step 2), every Day-1 pilot run.

**Mocked loop** (built/run in a prior pass):
`.claude_z4b_tmp/step7_mocked_loop.py`, results in
`.claude_z4b_tmp/step7_results.log` (this worktree). Covers rectangle SJ,
triangle-35 SJ, vessel plan B/B' -- periods 100.57-100.62ms, counter-mode
confirmed, zero deadline misses. **INV-7 not covered (open gap, above).**

## Section E -- K2: real arm, no MCR (full runnable procedure)

### E1. Pre-motion setup confirmation (before any motion)

```bash
# Read back the measured TCP box the campaign will run against
python3 -c "
from proper_research.hardware.online.rectangle_stage_a import common
print('min_m:', common.WORKSPACE_XYZ_MIN_M)
print('max_m:', common.WORKSPACE_XYZ_MAX_M)
"
# Expect: min_m=(0.24469, -0.89361, 0.36544), max_m=(0.58971, -0.49576, 0.46496)
```

Physically confirm against the rig per the Round 15/R1 measurement
procedure already in this repo's history (the box is the author's own
jogged physical limits minus a 20mm margin; this session found no
separate standalone document repeating that exact measurement procedure
beyond the `common.py` comment trail itself -- read `common.py`'s header
comments directly for the full provenance chain of these six numbers
before K2, rather than re-deriving them).

### E2. Reduced-speed extreme-pose clearance check -- MANUAL PROCEDURE, not a code command

Validates the closed TCP-box decision above. **No safe, collision-checked
route from the robot's actual current state to an arbitrary extreme pose
exists anywhere in this codebase.** Checked: `robot_class.py`'s `URRtde`
exposes only `moveJ`/`moveL` (UR's own joint-space-interpolated motion --
no environment-aware collision checking) and `servoJ` (raw per-tick
streaming). `close_loop_path_follow.py` -- the shared loop every real
runner uses -- contains no `moveJ`/`moveL` call at all; it starts
`servoJ`-streaming immediately from tick 0 relative to whatever state the
robot is actually in. There is no staged, intermediate-sample-by-sample
route built anywhere: `run_open_loop_c.py` plays the FULL plan trajectory
via `servoJ`, with no flag to start mid-path, and it still begins from
tick 0 relative to the robot's current (not necessarily known-safe) state.
A direct `moveJ` straight to an extreme pose -- the only mechanism that
actually exists for "go to this specific pose" -- is validated only for
joint continuity, not for clearance from the table, camera mount, cables,
or the MCR hardware. **Forcing a code command here would not actually be
safe; this is the honest answer, not a gap to paper over.**

**Manual procedure instead -- review with the lab operator in person
before K2, do not execute unsupervised:**

1. Confirm the arm is at a known, visually-clear starting configuration
   (e.g. its normal rest/home pose) before touching any of this.
2. Operator stands at the robot with a hand on the e-stop. Jog the arm
   manually (teach pendant, freedrive, or `moveJ` at very low speed in
   small increments -- operator's own judgment call on method) toward
   each target pose below, pausing every small step to visually confirm
   clearance before continuing. Never command the full joint delta in one
   motion.
3. Target poses (sample index and joint angles, computed today from each
   rebuilt plan's own `time_parameterized_configuration_path.npz` `state`
   array -- for the operator's reference display, not for a one-shot
   command):

   | Plan | Pose | Sample idx | q (rad) | Clearance |
   |---|---|---|---|---|
   | Rectangle | start (z-tightest) | 0 of 213 | `[-0.8533, -1.8744, -1.7319, -1.1345, 1.5626, -2.0535]` | 24.73mm (z) |
   | Rectangle | end (x/y-tightest) | 213 of 213 | `[-0.6319, -1.8001, -1.8014, -1.2518, 1.6175, -2.0781]` | 80.25mm (x) / 85.76mm (y) |
   | Triangle-35 | start (z-tightest) | 0 of 411 | same as rectangle start | 24.73mm (z) |
   | Triangle-35 | x-tightest | 404 of 411 | `[-0.9794, -1.8521, -1.7299, -1.3204, 1.3433, -2.0543]` | 26.45mm (x) |
   | Triangle-35 | y-tightest | 375 of 411 | `[-1.0112, -1.9943, -1.5782, -1.1157, 1.2898, -1.9891]` | 26.26mm (y) |

4. **Abort criteria (stop immediately, release e-stop):** any visible
   contact or near-contact with the table, camera/mount, cables, or MCR
   hardware before reaching the target; any unexpected joint behavior;
   any doubt at all about clearance.
5. Record, per pose reached: whether clearance looked as expected given
   the table above, and a photo if practical, in the K2 run notes.
6. Return the arm to its home/rest pose manually when done, the same way.

### E3. N=15 streaming timing check

**No schedule cache exists yet for the z4b rectangle plan** -- build it
for real first (this calls `common.build_or_load_schedule`, the one Z4
consolidated schedule builder; takes a few minutes, pure computation, no
robot):

```bash
python3 -c "
from proper_research.hardware.online.rectangle_stage_a import common
common.build_or_load_schedule(
    'plans/rectangle_10x15mm_skipglobal_z4b_2026-11-11/time_parameterized_configuration_path',
    'plans/rectangle_10x15mm_skipglobal_z4b_2026-11-11/jacobian_schedule_insertion_anchor.npy',
)
"
```

```bash
# MOVES THE ARM. Rectangle MPC-SJ N=15, as in Section C, with streaming
# on by default (Z4b step 1; confirm via controller_metadata.json["progress_mode"]=="counter").
# --run-name makes this K2 check's output directory collision-proof
# against any real campaign run (its default would be a fixed, reusable
# name -- this run-name is unique to this K2 check).
python3 -m proper_research.hardware.online.rectangle_stage_a.run_mpc_delay_aware_insertion_anchor \
  --plan-dir plans/rectangle_10x15mm_skipglobal_z4b_2026-11-11/time_parameterized_configuration_path \
  --out-dir close_loop_logs/myrun \
  --schedule-cache plans/rectangle_10x15mm_skipglobal_z4b_2026-11-11/jacobian_schedule_insertion_anchor.npy \
  --run-name k2_timing_check_$(date -u +%Y%m%dT%H%M%SZ) \
  --horizon 15 --max-tracking-error-mm 7

# Extract and summarize from the resulting path_follow.jsonl:
**Field verification (2026-11-11): `path_follow.jsonl` has TWO timing
fields, not one.** `close_loop_path_follow.py`'s own row-dict code
(where this file is actually written) shows `t_s` is computed near the
END of each tick's work (`t_rel = now_monotonic() - t_zero`, right before
the row is assembled), while `loop_start_s` is captured "at the top of
this tick, right after the scheduler advanced next_tick -- the true
start of this tick's own work, independent of servo_j()'s blocking"
(the code's own Round-7 comment). **`loop_start_s` is the correct field
for measuring the scheduler's realized PERIOD** -- `t_s` includes
variable per-tick processing time and is not what Round 7's timing fix
was built to validate. Also verified: `q_cmd_k`/`q_cmd_km1` (used in an
earlier pass of this guide) **do not exist in `path_follow.jsonl` at
all** -- those belong to a different class's separate internal log
(`process_isolated_adapter.py`'s `ProcessIsolatedBaselineAdapter`, not
the one these runners use). The real, verified commanded-step field is
`q_target_delta_rad` (this tick's own commanded joint delta, logged in
the SAME row, no cross-row diff needed).

**Pairing check (confirmed, not assumed):** under this loop's own
"measure current state, then compute/apply this tick's command, then
log" order, row k's `q_meas_rad` is the state BEFORE row k's command is
applied. So the step actually realized by command k is
`q_meas_rad[k+1] - q_meas_rad[k]`, correctly paired against THIS SAME
row's `q_target_delta_rad[k]` (same tick index k) -- not an off-by-one;
confirmed by the field semantics above, not just asserted.

```bash
python3 -c "
import json, numpy as np
rows = [json.loads(l) for l in open('<the new K2 run dir>/path_follow.jsonl')]
loop_start = np.array([r['loop_start_s'] for r in rows])
period = np.diff(loop_start)
print('period (ms) mean/median/max:', np.mean(period)*1e3, np.median(period)*1e3, np.max(period)*1e3)
misses = sum(1 for r in rows if r.get('deadline_miss'))
print('deadline misses:', misses, '/', len(rows))
latency = [r['target_update_ms'] for r in rows if r.get('target_update_ms') is not None]
print('streamer pickup latency (ms) mean/max:', np.mean(latency), np.max(latency))
q_meas = np.array([r['q_meas_rad'] for r in rows])
q_delta = np.array([r['q_target_delta_rad'] for r in rows])
n = min(len(q_meas) - 1, len(q_delta))
realized = np.diff(q_meas, axis=0)[:n]
commanded = q_delta[:n]
ratio = np.abs(realized) / np.maximum(np.abs(commanded), 1e-12)
print('median realized/commanded ratio per joint:', np.median(ratio, axis=0))
print('target: median >= 0.95 for every joint')
"
```
Fields used, all confirmed real: `loop_start_s` (period), `deadline_miss`
(misses), `target_update_ms` (streamer pickup latency), `q_target_delta_rad`
(commanded step, same row) vs. `q_meas_rad` diffed across rows (realized
step), all 6 joints, correctly paired by tick index.
```

### E4. Five-fold delay re-identification at d=1, 2, 3 -- **STOP CONDITION, read this first**

> **A result other than d=2 is a STOP condition.** Do not proceed to
> Day-1 pilots or the core campaign if re-identification gives a
> different d. The delay model changes, and every hardware condition
> must be re-run under the new d before mixing old and new data.

**A new, complete 5-fold leave-one-out procedure now exists** --
`proper_research/studies/mpc_ablations/delay_identification_5fold.py`. (The original fitting
script and its two `.pkl` inputs are confirmed permanently gone --
`SUPPLEMENTARY_EVIDENCE.md` S10 says the script "is no longer present"
and this session confirmed `delay_model_results.pkl`/`tau_loo.pkl` exist
nowhere in this checkout; only the markdown table survives. Per the
author's decision, this is resolved by identifying d FRESH from K2 data
rather than chasing the unrecoverable original, using
`study05_delay_model_sweep.py`'s single-log replay NOT substituted in
its place.)

**Inputs:** one run's `path_follow.jsonl` with >=150 ticks (5 folds need
>=30 ticks each for a stable estimate) -- the K2 N=15 timing-check run
(E3 above) is the natural source, since it's already being collected.
**Fields:** `q_target_delta_rad` (commanded delta, same row) and
`q_meas_rad` (measured joints) -- the same two fields E3's fix above
verified, with the same tick-pairing.

**Fold definition:** the tick sequence is split into 5 CONTIGUOUS,
equal-length blocks (not a random shuffle -- this is a time series;
shuffling would let neighbouring ticks leak across the train/held-out
split and bias the result toward noise). Each fold is held out once;
the other 4 are pooled as train.

**What's computed:** for each candidate delay d in {0,1,2,3}, predict
the realized step at tick k as the commanded delta from d ticks earlier
(`q_target_delta_rad[k-d]`), and compute RMSE against the actual
realized step (`q_meas_rad[k+1]-q_meas_rad[k]`), on train and held-out
ticks separately. d=0 is the no-delay baseline. There is nothing to
numerically fit beyond the integer d itself -- the held-out RMSE is what
the 5-fold structure actually buys over a single-log check.
$G_d = 1-(\text{RMSE}_{\text{heldout},d}/\text{RMSE}_{\text{heldout},0})^2$
per fold, matching S10's own formula, so a result is comparable to the
historical (now-unreproducible) number if wanted, without depending on
anything from that original run.

> **Stop rule.** CONFIRMED (d=2): d=2 gives the lowest held-out RMSE on
> >=4 of 5 folds, AND $G_2>0.9$ on every fold. **STOP** otherwise (fold
> disagreement, or $G_2\le0.9$ on any fold) -- investigate before Day-1
> or the core campaign, per the same d≠2-is-a-stop-condition rule as E4.

```bash
python3 proper_research/studies/mpc_ablations/delay_identification_5fold.py \
  --log <the new K2 run dir>/path_follow.jsonl
```

**Smoke-tested 2026-11-11** against one real historical vessel log
(`close_loop_logs/myrun/mpc_closedloop_contact_floor240_dipolealign_recal_matchedtarget_dipolealigned_20261020T131642Z/path_follow.jsonl`)
to confirm the script runs correctly end to end: it did, and produced a
real (STOP: best d=1 on all 5 folds, $G_2$ 0.20-0.36) result -- **this is
a script-correctness check only, not a finding to act on.** That log
predates the Z4b timing fix (its own period averaged 126.6ms, not the
fixed 100.0ms) and is a vessel-rig log, not the rectangle K2 context the
original d=2 identification targeted -- running on the wrong plant/timing
regime is expected to give a different answer. **Run this for real only
on a fresh K2 log**, after the timing fix and on the actual condition
being validated.

### E5. Pre-flight refusal test

```bash
# Entirely rooted in /home/jack/Proper-Research -- no worktree path, no
# robot/camera import. Re-run live 2026-11-11 to confirm: PASSED, all 12
# deliberately-wrong-field-group cases refused, correct config clean.
cd /home/jack/Proper-Research
python3 proper_research/hardware/online/test_stage_a_common.py
```

Verify required log fields are present after a real K2 run:

```bash
python3 -c "
import json
row = json.loads(open('<run dir>/path_follow.jsonl').readline())
for f in ['loop_start_s','round_trip_ms','deadline_miss','qp_status','argv']:
    print(f, '->', f in row or f in row.get('controller_metadata', {}))
"
```

## Section F -- Day-1 pilots + go/no-go

**Real lab-plan pilot set (not this session's earlier guess):** three
repetitions each of vessel SJ N=15, vessel SJ N=3, and rectangle SJ N=15 --
**9 runs total, no INV-7 in the pilot.**

**Go/no-go criteria.** Checkable now (no numbers needed):

- No unexpected stops: `summary.json["stop_reason"] == "path_complete"`
  for every one of the 9 runs.
- Complete logs for every run: `path_follow.jsonl`,
  `controller_metadata.json` (or `summary.json["config"]` for scripts
  without it), and `tip_trajectory.csv` all present and non-empty.

**Numeric thresholds -- PROPOSED, needs the author's approval, not a
settled criterion.** Pulled from one real historical vessel plan-B log
(`close_loop_logs/myrun/mpc_closedloop_contact_floor240_dipolealign_recal_matchedtarget_dipolealigned_20261020T131642Z/path_follow.jsonl`,
re-read directly 2026-11-11):
- **Period:** that log's own period was mean 126.6ms / median 121.1ms /
  max 171.4ms -- **but this log predates the Z4b streaming/timing fix**
  (the fix that gives the 100.0ms period confirmed in K1/K2), so these
  numbers describe the OLD system, not a target for the new one. Proposed
  threshold: period mean within 5ms of 100ms (i.e. 95-105ms), based on
  the fix's own K1 result (100.0ms, 4.7ms mean / 5.1ms max pickup
  latency) -- the author should confirm or replace this band.
- **Joint servo-tracking error (closest real proxy for "drift" found this
  session; not identical to delay-aligned joint divergence):** that same
  log's `q_servo_error_rad` field, all 6 joints, gave max
  0.49-0.74\textdegree{} per joint, mean 0.08-0.17\textdegree{}, final-tick
  0.02-0.16\textdegree{} -- a completed, non-aborted run. Proposed
  threshold: max per-joint servo error under ~1\textdegree{} and not
  trending upward over the run's second half vs. its first -- the author
  should confirm this is the right field/threshold, since it measures the
  robot's own servo tracking, not necessarily the "delay-aligned joint
  drift" concept as originally intended.
- **Insertion tracking around reversals:** no historical log analysis was
  done for this specific question this session -- no number proposed;
  flagged as open, not guessed at.
- No unexpected stops (checkable now: `summary.json["stop_reason"] ==
  "path_complete"` for every one of the 9 runs).
- Complete logs for every run (checkable now: `path_follow.jsonl`,
  `controller_metadata.json` or `summary.json["config"]`, and
  `tip_trajectory.csv` all present and non-empty for every run).

## Section G -- Later phases (planned, not dropped)

**Trap paths** (campaign-order step 5): needs path design (speed fraction
`alpha`, planned clearance `delta`), an L1-L3 build, the resulting plan
file + SHA-256, and a simulation screen (per the plan document's K3
section) before any hardware run. No settings invented here.

**Silicone demonstration** (campaign-order step 6, final): the author's
own scoping work -- phantom material/fabrication, centreline/lumen-radius
extraction, an L1-L3 build, its own protocol entry. Not part of K2 or the
core campaign.

## Checklist A -- No-robot preparation (do all of this before connecting to the arm)

1. Read back the TCP box constants and confirm against the physical rig (Section E1).
2. Build the rectangle schedule cache fresh (Section E2/C):
   `python3 -c "from proper_research.hardware.online.rectangle_stage_a import common; common.build_or_load_schedule('plans/rectangle_10x15mm_skipglobal_z4b_2026-11-11/time_parameterized_configuration_path', 'plans/rectangle_10x15mm_skipglobal_z4b_2026-11-11/jacobian_schedule_insertion_anchor.npy')"`
3. Hash-check both rebuilt campaign plans against the protocol's recorded SHA-256 (Section A).
4. Run the pre-flight refusal test and confirm PASS: `cd /home/jack/Proper-Research && python3 proper_research/hardware/online/test_stage_a_common.py`.
5. Confirm `protocol_campaign2.yaml`'s `status` still reads `DRAFT` (it has not been reviewed/signed off).
6. Review the manual teach-pendant pose-check procedure (Section E2) with the lab operator in person, before going to the lab.

## Checklist B -- K2 session (in the lab, with the lab operator)

1. **Manual teach-pendant pose checks** (Section E2, settled): operator-supervised, incremental jogging to each extreme pose in the table, visual clearance confirmation, abort on any doubt.
2. **N=15 timing-check run** (Section E3): run the fixed command (reads `loop_start_s` for period, `target_update_ms` for pickup latency, `deadline_miss` for misses); confirm period near 100ms and zero/near-zero deadline misses.
3. **Step-realization check** on that same run's log, using the corrected, verified fields (`q_target_delta_rad` vs. `q_meas_rad`, same-tick pairing): confirm median realized/commanded ratio >= 0.95 on every one of the 6 joints.
4. **Five-fold delay identification** on that same K2 log: `python3 proper_research/studies/mpc_ablations/delay_identification_5fold.py --log <the K2 run dir>/path_follow.jsonl`. **Apply the stop rule: d=2 confirmed only if >=4/5 folds agree and G_2 > 0.9 on every fold; otherwise STOP and investigate before Day-1.**
5. Confirm the required log fields are present (`loop_start_s`, `target_update_ms`, `deadline_miss`, argv, stationary start-up frames) in the resulting run directory.
6. Only if 1-5 all pass: proceed to Day-1 pilots (Section F) -- not before.

## Synthetic validation of the 5-fold delay-identification script (2026-11-11)

**What this proves, and what it doesn't.** The checks below confirm the
script's fold/stop-rule logic correctly recovers an ALREADY-KNOWN,
injected delay under controlled conditions. **They do not, by themselves,
prove the real hardware's delay is still d=2** -- that still requires
running the script on real fresh K2 data (Checklist B, item 4). Treat
this as validating the tool, not as a substitute for the on-hardware check.

**Generator:** `.claude_z4b_tmp/gen_synthetic_delay_logs.py` -- builds three
300-tick, 6-joint synthetic logs (`q_meas_rad`, `q_target_delta_rad`, the
same two real fields the script uses), each with a KNOWN injected delay
(realized step at tick k = the commanded delta from exactly d_true ticks
earlier, plus small Gaussian noise, std 1e-5 rad) for d_true in {1, 2, 3}.

```bash
python3 .claude_z4b_tmp/gen_synthetic_delay_logs.py
for d in 1 2 3; do
  echo "===== synthetic d_true=$d ====="
  python3 proper_research/studies/mpc_ablations/delay_identification_5fold.py \
    --log .claude_z4b_tmp/synthetic_d${d}_path_follow.jsonl \
    2>&1 | grep -E "best d on this|vote counts|RESULT:|G_[0-9] per fold"
done
```

**Real output, this session:**

| d_true | vote counts | G_2 per fold | Result |
|---|---|---|---|
| 1 | `{1: 5}` | 0.40, 0.00, -0.00, 0.00, 0.00 | STOP (correct -- d=2 correctly rejected) |
| 2 | `{2: 5}` | 1.00, 1.00, 1.00, 1.00, 1.00 | CONFIRMED (correct) |
| 3 | `{3: 5}` | 0.84, 0.89, 0.89, 0.89, 0.89 | STOP (correct -- notably, G_2 is substantial here since d=3 is structurally close to d=2, but still correctly below the 0.9 bar, so the stop rule does NOT falsely confirm) |

All three recovered the true injected delay exactly, with unanimous
(5/5) fold agreement in every case, and the mismatched cases (d_true=1,3)
correctly failed to confirm d=2 -- including the harder d_true=3 case,
where a naive "G_2 looks okay" read could have been misleading without
the explicit >0.9 threshold.

## Step-realization threshold: per-joint or aggregate? (resolved, with source quote)

Checked the plan document (`mcr-analysis-plan-batch2*.md`, the Round 34
source used throughout this guide) directly. Three mentions, none fully
disambiguating:
- Round 7/J2: "the arm still realizes **≥95% of each step**."
- Round 8/K2 item 2: "realized/commanded joint step per tick; **target
  ≥0.95 median, as before**."
- A later round: "Step realization, median ≥0.95."

**Genuinely ambiguous on close reading** -- none of the three ever states
whether the ratio is computed per joint (6 separate medians, each needing
≥0.95) or pooled across all joint-tick pairs into one number. "each
step" (not "each joint's step") reads slightly toward a per-tick, possibly
already-aggregated quantity; "joint step" (singular) in the K2 line reads
slightly toward per-joint. Neither phrase settles it.

**Recommendation (this session's judgment call, not the source document's):
per-joint, not pooled.** A pooled ratio could pass at ≥0.95 even if one
joint is consistently under-realized, as long as the other five compensate
in the average -- exactly the kind of masked single-joint fault a safety
check before K2/Day-1 should not let through. The guide's calculation
(Section E3, Checklist B item 3) already computes and reports the median
ratio **per joint** (`np.median(ratio, axis=0)`, printing all 6 values) and
requires every one of the 6 to clear 0.95 -- this is the stricter, safer
reading, kept as the working criterion. **If the author intends the pooled
reading instead, say so explicitly** -- the aggregate number is trivially
available too (`np.median(ratio)` over the flattened array) but was not
made the primary criterion here without a clearer source instruction.
