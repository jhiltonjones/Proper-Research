# Advancer-excitation identification test

One question: **does smooth commanded insertion produce real beam-tip
oscillation that the feedback controller later chases?** This is the
"missing plant-model state" line of investigation from the triangle MPC
hidden-state work -- the advancer has no encoder, its rate is fed into
neither controller's model, and if it excites a vibration mode the
controller only sees through vision, the controller cannot distinguish
"tracking error caused by my own lag" from "tracking error caused by a beam
mode I'm now chasing." This test isolates the physical disturbance at its
source, with the robot fixed, before drawing any conclusion from a
closed-loop run.

Run this **before** another closed-loop robot run that would otherwise be
interpreted as evidence about controller quality -- see the predeclared
decision rule at the bottom.

## Pipeline

```
acquire.py (LIVE, robot fixed)  ->  analyze.py (offline)  ->  causality.py (offline)
```

```bash
# 1. Live acquisition. Robot is NEVER commanded by this script -- fix it (and
#    the magnet) in a safe pose yourself first, or pass --move-to-plan-initial
#    for a one-shot moveJ that releases the connection before the test starts.
#    Default is a DRY RUN (sequencing/camera/logging exercised, nothing
#    moves); pass --live to actually drive the advancer.
python -m proper_research.hardware.online.advancer_excitation.acquire \
    --plan-dir plans/rectangle_10x15mm_bimaterial/time_parameterized_configuration_path \
    --l0-mm 25.0 \
    --out-dir advancer_excitation_logs/rectangle_2026-09-18 \
    --move-to-plan-initial --live

# 2. Offline ring-down / PSD / G_excitation analysis + trend fits, and a
#    resampled-at-10Hz vibration record per length for step 3.
python -m proper_research.hardware.online.advancer_excitation.analyze \
    --log advancer_excitation_logs/rectangle_2026-09-18/advancer_excitation_<stamp>.jsonl \
    --out advancer_excitation_logs/rectangle_2026-09-18/analysis.npz

# 3. Offline causality: feed the measured vibration through INV's own J^+kp
#    mapping (kp sweep), and spectrally compare against an existing
#    closed-loop run.
python -m proper_research.hardware.online.advancer_excitation.causality \
    --analysis advancer_excitation_logs/rectangle_2026-09-18/analysis.npz \
    --L-index 4 \
    --plan-dir plans/rectangle_10x15mm_bimaterial/time_parameterized_configuration_path \
    --closed-loop-run close_loop_logs/rectangle_inv_2dof_trim_2026-09-17/inv_2dof_trim_accumC_kn0_kp0.6_20260917T175242Z
```

## Protocol (as specified, implemented as-is)

Four insertion lengths spanning the plan's own range, `L_i = L_min + i/3 *
(L_max-L_min)` for i=0,1,2,3 (`common.length_plan_from_plan_dir` reads
`L_min`/`L_max`/the representative insertion rate straight from the plan's
`state`/`input_reference` arrays -- never hand-copied). At each length: one
~5s stationary baseline, then 5 repeats of `2s hold -> advance 1.5mm at the
plan's own representative rate -> 3s hold (ring-down) -> retract back to
`L_i``. Camera frames are logged as they arrive (native reconstruction
rate, `acquire._run_phase` busy-polls the state slot's sequence number) --
never downsampled to the controller's 10Hz during acquisition.

## Metrics (analyze.py)

Per repeat, over the post-advance hold: `p_inf` = mean tip position over the
final 0.5s, `r(t) = p(t) - p_inf`, `r_3D(t) = ||r(t)||`.

- `A_peak` = max `||r(t)||` over the first 1s after the advancer stops.
- `R_ring` = RMS of `||r(t)||` over that same 1s.
- `R_noise` = RMS of `p - mean(p)` over the stationary baseline at that length.
- `G_excitation = R_ring / R_noise`.
- `f_osc` = dominant PSD frequency, **not** of `r_3D` -- see the bug below.
- `zeta` = damping ratio from the log-decrement of consecutive ring-down peaks.
- `t_10%` = first time after which `||r(t)||` stays below `0.1*A_peak`.

`A_peak(L)` and `f_osc(L)` are fit linearly across the four lengths and
reported with R².

### A real bug caught on synthetic ground-truth data: rectifying `r_3D` doubles the apparent frequency and halves the apparent damping ratio

`r_3D(t) = ||r(t)||` is always non-negative. For a beam oscillating
predominantly along one direction (the expected case -- bending in a
single plane), `r_3D` is the *rectified* magnitude of a signed oscillation:
it has two lobes per true cycle, not one. Feeding it straight into an FFT
folds the true frequency onto **2x** its real value, and picking successive
`r_3D` peaks for the log-decrement compares amplitudes **half a period**
apart instead of a full period, which roughly **halves** the recovered
`zeta` (both confirmed on synthetic data with known injected `f_osc`/`zeta`:
recovered values were off by almost exactly 2x/0.5x before the fix, and
matched to <1% after).

Fix (already applied): `f_osc` and `zeta` are computed from the residual's
projection onto its dominant PCA direction (`_dominant_signed_component`,
sign-preserving) instead of `r_3D`. `A_peak`/`R_ring` correctly keep using
`r_3D` as specified -- those are amplitude metrics where rectification
doesn't matter.

A second, smaller fix: the PSD window is capped at `psd_window_s` (2s, not
the full hold-post window) -- for a fast-decaying mode the long noise-only
tail otherwise dilutes the transient's SNR badly enough to shift the
detected peak. Verify both fixes still look right on the real data (the
synthetic test used a clean single-mode decay; a real beam may have several
modes close together).

## Part 1: does the vibration alone reproduce the live jagged-command signature?

`causality.py` replays the measured vibration (resampled to 10Hz) through
**the exact damped pseudoinverse `TwoDOFTrimController` uses**
(`common.damped_pinv_joint_jacobian`, `damping=5e-2`, joint-only 3x6 block)
at a single fixed Jacobian evaluated at the test pose (robot state is fixed
throughout, so one evaluation is correct, not an approximation):

```
dz_vib,k = -J^+ (kp * r_k)
```

for `kp` in `{0.3, 0.6, 1.0}`. Reports `R_q` (RMS trim magnitude), the
fraction of ticks that would trip the rate bound
(`rectangle_stage_a.common.MAX_JOINT_STEP_RAD = 0.010 rad`), peak joint
increment, and the dominant frequency of the resulting command. The
decisive pattern: `R_q(kp=1.0) > R_q(kp=0.6) > R_q(kp=0.3)` (this part is
guaranteed by construction -- the map is linear in `kp` -- so it is a sanity
check, not evidence) **and** the command's dominant frequency landing near
the excitation test's own `f_osc` for that length (this part is the real
test).

## Part 2: spectral coherence against an existing closed-loop run

From `path_follow.jsonl`: PSD of `error_norm_mm` (the tracking residual)
and of `||u0[:6]||` (the feedback command), plus their cross-correlation at
lag 0/1/2 ticks. Compared against the excitation test's `f_osc` for
whichever length's `r_k` was used in Part 1.

Note on notation: the test's original specification writes `r_k` for both
"the excitation-test vibration residual" and "the closed-loop tracking
residual" -- they are **different signals**. Part 1 uses the former (from
`acquire.py`, robot fixed, advancer-only). Part 2's `e(t)`/cross-correlation
uses the latter (from a real closed-loop run, both the beam and the
controller are moving). `causality.py` keeps them as separate variables
(`r_k` vs `e_norm`) to avoid the ambiguity.

## The predeclared decision rule

**Strongly supported** if most of these hold: `G_excitation >> 1` (several
times the noise floor); `A_peak`/`R_ring` increase materially with `L`;
`f_osc` is repeatable across repeats at a length; that same `f_osc` shows up
in Part 2's closed-loop spectra; and Part 1's `R_q` grows with `kp` (a given)
while its dominant frequency lands near `f_osc` (the actual test).

**Rejected** if the beam is essentially stationary right after the
advancer stops, `R_ring` ≈ `R_noise`, there's no length trend, and Part 1's
command barely responds to the measured residual.

## Safety notes

- `acquire.py` never opens a robot **control** connection and issues zero
  robot commands, live or dry-run -- only an optional receive-only
  `RobotJointStream` sanity log (best-effort; failure to connect does not
  block the test). Position the robot/magnet yourself, or use
  `--move-to-plan-initial` for a one-shot reset that releases its
  connection before the sequencing loop starts.
- The advancer only moves with `--live`. Without it, the full sequence
  (camera, logging, advancer integration, timing) runs at full speed with
  `AdvancerSinkConfig(dry_run=True)` -- no serial port is opened.
- `analyze.py` and `causality.py` touch no hardware at all.
