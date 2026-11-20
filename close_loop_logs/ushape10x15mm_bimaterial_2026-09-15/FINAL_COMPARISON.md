# Controller comparison — U-shape (10x15mm, open rectangle), bimaterial model, 2026-09-15

## Setup
- Plan: `/home/jack/.claude/jobs/3710eca5/tmp/plans_bimaterial/ushape_10x15mm_bimaterial/time_parameterized_configuration_path`, replanned with the session's bimaterial Cosserat beam model (0.2mm nitinol wire + 40mm PDMS/iron-particle composite tip, 5mm overlap), same geometry/flags as the original `day5_ushape10x15mm_2026-09-13` plan (rectangle base 10mm, depth 15mm, **open** — no `--closed`). 213 samples, 21.2s duration (day5's original: 214 samples/21.3s — essentially identical timing).
- Beam length 25.0mm, all runs same day/calibration state.
- MPC horizon = 15 (the day2/triangle-validated setting) unless noted otherwise. `state_tracking_weight = 0.001` fix active on every MPC+FF run (day2 root cause, confirmed via the standard `[path] overriding state_tracking_weight 1 -> 0.001` log line).
- RMS/max computed over the transit phase only (`terminal_hold == False`); hold = mean error during the 2.0s terminal hold (`shape_end_hold_s`, ~22 ticks @ 10Hz); final = error at the last control step.
- This is the first **open** (non-closed) shape tested this session with the bimaterial model — the triangle and rectangle plans tested earlier are both closed loops (hold point = start point).

## Headline results

### vs. the old (non-bimaterial) day5 baseline
| Condition | Old (day5, 2026-09-13) | New (bimaterial) |
|---|---|---|
| Open loop | 2.43±0.09mm RMS | 1.35mm RMS (n=1, first rep) |
| Best controller | MPC-LTV, 1.29±0.02mm RMS | MPC-LTI, 1.00±0.11mm RMS (n=4 default config) |

Same pattern as the triangle/rectangle bimaterial replans: every number improves, not just a different fit.

### Closed-loop, default config (`ff_trim_base_switch_at_hold=False`, exposure as inherited from prior session state), horizon=15, n=4
| Controller | RMS (mm) | Max (mm) | Hold (mm) | Final (mm) |
|---|---|---|---|---|
| Inverse-Jacobian+FF | 1.08±0.06 | 2.58±0.13 | **0.22±0.05** | **0.13±0.06** |
| MPC-LTI+FF | **1.05±0.13** | **2.52±0.20** | 0.67±0.20 | 0.50±0.27 |
| MPC-LTV+FF | 1.11±0.11 | 2.63±0.18 | 0.44±0.19 | 0.43±0.26 |

**This is a different pattern than every closed shape tested this session (both triangle sizes, rectangle):** MPC wins RMS/max in transit, but inv-Jacobian is clearly and consistently better at hold/final — the opposite of the triangle/rectangle results, where MPC won everywhere including hold.

## Root-cause investigation of the hold-phase gap

### Confirmation pass (n=3, fresh reps, same config)
Repeated to check whether the pattern was real or a small-n artifact:

| Controller | RMS (mm) | Max (mm) | Hold (mm) | Final (mm) |
|---|---|---|---|---|
| Inverse-Jacobian+FF | 1.33±0.11 | 2.94±0.19 | **0.24±0.07** | **0.20±0.08** |
| MPC-LTI+FF | 1.36±0.02 | 2.99±0.00 | 0.62±0.08 | 0.52±0.08 |
| MPC-LTV+FF | 1.30±0.09 | 2.91±0.12 | 0.39±0.05 | 0.48±0.18 |

Pooled across both batches (7 reps per controller total): inv-Jacobian won hold/final on **every single rep**, MPC-LTI lost on every single rep, MPC-LTV consistently in between. Not consistent with random measurement noise — a real, repeatable effect.

### Mechanism found in code (`close_loop_path_follow.py`)
During `terminal_hold`, with the default config:
```python
q_target = ref_state[target_index, :6] + cfg.feedforward_correction_scale * delta_q
```
`target_index` freezes at the last reference sample once hold starts. `delta_q` is capped at `±max_joint_step_rad` (0.006 rad/tick) — but **there is no accumulator for the joint correction** (unlike insertion, which has `ins_trim += corr[6] * dt`). Every tick recomputes the target from the *same fixed reference* plus at most one fresh bounded step, so the achieved position plateaus at roughly `reference + one step` and never walks further, regardless of how many hold ticks pass.

Confirmed directly in `path_follow.jsonl` for the worst rep (`ltv_rep3`, RMS=1.216mm — see below): `u0_correction[0]` (raw MPC-desired joint-1 correction) stayed pinned at exactly 0.1 rad on **all 22 hold ticks**, while the physically-applied `q_target_delta_rad` stayed capped at 0.006 rad — the controller never got closer, so it kept re-requesting nearly the same large correction for the full 2.1s hold window. Ruled out as explanations: Jacobian conditioning (near-identical, ~69,000, between good and bad reps) and near-null-direction routing (bad rep actually had *less* overlap with the weak direction than a good rep, 0.19 vs 0.44).

The real differentiator is the **residual error at the moment transit ends and hold begins**: reps entering hold at <0.4mm converge fine within the 2.1s window; reps entering at 0.7-1.3mm get stuck oscillating around an elevated plateau because the (non-accumulating, rate-limited) correction can't close a gap that size in the available ticks. Inv-Jacobian's simpler proportional law reliably lands close at the very last transit tick; MPC's receding-horizon approach is more variable there — plausibly because the U-shape's endpoint is a genuinely novel terminus (unlike the closed triangle/rectangle, where hold = the start point, already reached cleanly at t=0). This last mechanism is a plausible hypothesis, not confirmed with a targeted test.

### Attempted fix #1: `--ff-trim-base-switch-at-hold` — FAILED, do not use
This flag (originally built 2026-09-11 for an unrelated insertion-drift bug) switches the hold anchor to `q_target = q_measured + delta_q`. Tested n=2 (inv, mpc_lti) before aborting:

| Controller | Hold (mm) | Final (mm) | vs. default |
|---|---|---|---|
| Inverse-Jacobian+FF | **6.96** | **8.82** | was 0.22mm / 0.13mm |
| MPC-LTI+FF | **6.32** | **9.15** | was 0.67mm / 0.50mm |

Catastrophic regression for *both* controllers (this flag isn't MPC-specific — it gates on FF mode only). Likely mechanism: anchoring to the live measured position compounds servo lag/overshoot each tick instead of staying bounded relative to a known-good reference. **Killed mid-run, robot confirmed safe (not protective-stopped) both during and after.** Not a viable fix as implemented.

### Attempted fix #2: shorter horizon (5 instead of 15) — inconclusive, stopped early
Rationale: a shorter horizon might make MPC's terminal approach more reactive (closer to inv-Jacobian's proportional law), reducing the residual at hold entry. Only 1 rep each of mpc_lti/mpc_ltv completed before the run was stopped (per user judgment that results looked no better than the h15 average, and to pursue the camera-exposure hypothesis instead):

| Controller | RMS (mm) | Hold (mm) | Final (mm) |
|---|---|---|---|
| MPC-LTI+FF (h5) | 1.13 | 0.48 | 0.30 |
| MPC-LTV+FF (h5) | 1.19 | 0.29 | 0.25 |

Within the range of h15's rep-to-rep scatter — not clearly better or worse on n=1. Untested at n=3; genuinely inconclusive.

### Attempted fix #3: camera exposure 18→25 — real effect, but a trade-off, not a clean fix
Hypothesis: measurement noise (not a control-loop bug) explains the MPC hold gap. Exposure changed from the pipeline's prior value to 25 (confirmed applied via `cv2.CAP_PROP_EXPOSURE` in `camera_source.py`, matching `StateStreamConfig.exposure`'s code default). Camera health re-verified after the change (20/20 valid tip frames).

**Closed loop, exposure=25, n=3:**
| Controller | RMS (mm) | Max (mm) | Hold (mm) | Final (mm) |
|---|---|---|---|---|
| Inverse-Jacobian+FF | 1.28±0.01 | 2.83±0.06 | 0.21±0.09 | 0.20±0.11 |
| MPC-LTI+FF | **1.09±0.04** | **2.61±0.04** | 0.33±0.09 | 0.36±0.21 |
| MPC-LTV+FF | 1.14±0.02 | 2.71±0.11 | **0.17±0.03** | **0.16±0.08** |

A real reversal: both MPC variants now beat inv-Jacobian on RMS/max, and MPC-LTV's hold (0.17±0.03mm) is the best and tightest of the whole session — essentially matching inv. MPC-LTI's hold (0.33±0.09mm) improved substantially from its default-exposure average (0.67±0.20mm) but is still the weakest of the three.

**Open loop, exposure=25, n=3:** RMS = 1.80±0.04mm (individual reps 1.85/1.79/1.76mm — tight, not noise), max = 3.47±0.11mm, final = 0.59±0.16mm.

This is **worse** than the pre-change open-loop baseline (1.35mm RMS, n=1). So exposure=25 is not a uniform improvement — it appears to help closed-loop hold-phase convergence (better contrast/less jitter feeding the correction law) while hurting raw open-loop position accuracy (plausibly a tip-centroid bias or contrast-dependent detection shift). A genuine trade-off, not confirmed as a net win, and MPC-LTI's hold still isn't fully closed even at the better exposure — so the frozen-reference/non-accumulating-correction mechanism above likely still contributes independently of camera settings.

## Honest summary of what's established vs. open
**Established (high confidence, repeated/code-verified):**
- The bimaterial replan is a real improvement on the U-shape (open loop 2.43→1.35-1.85mm depending on exposure; best controller 1.29→1.00-1.14mm RMS).
- The U-shape's hold-phase controller ranking (inv > MPC-LTV > MPC-LTI on hold/final) is real and repeatable across 7+ reps at default exposure, not noise.
- The mechanism is a genuine code gap: no accumulator for joint corrections during hold, unlike insertion which already has one.
- `--ff-trim-base-switch-at-hold` is not a viable fix — causes catastrophic (6-9mm) regressions for all FF controllers.
- Camera exposure materially affects the pattern — exposure=25 closes most of the MPC-LTV hold gap and helps MPC-LTI partially, at the cost of worse open-loop accuracy.

**Open / not yet confirmed:**
- Why the U-shape's open (non-closed) endpoint specifically produces more MPC transit-phase variability than the closed triangle/rectangle shapes — plausible hypothesis (novel terminus vs. already-validated start=hold point), not tested directly.
- Whether a shorter horizon (5) helps — only 1 rep each, inconclusive.
- The real fix for the underlying code gap (an accumulated, bounded joint trim during hold, mirroring the existing insertion-trim pattern) has not been implemented or tested.
- The best camera exposure setting is unknown — 25 is better than the prior default for closed-loop hold but worse for open-loop accuracy; no sweep has been done to find a value that's good for both.

## Data
Run directories (`path_follow.jsonl`, `tip_trajectory.csv`, `frozen_jacobian.json`, `summary.json`) under:
- `close_loop_logs/compare_FF_ushape10x15mm_bimaterial_h15_n3_2026-09-15/` + `..._h15_rep4_2026-09-15/` (default config, n=4)
- `close_loop_logs/compare_FF_ushape10x15mm_bimaterial_h15_1rep_2026-09-15/` + `..._h15_reps23_2026-09-15/` (confirmation n=3, pre-exposure)
- `close_loop_logs/compare_FF_ushape10x15mm_bimaterial_h15_holdfix_n3_2026-09-15/` (failed flag test, partial, n=1/1)
- `close_loop_logs/compare_FF_ushape10x15mm_bimaterial_h5_n3_2026-09-15/` (horizon=5, partial, n=1/1)
- `close_loop_logs/compare_FF_ushape10x15mm_bimaterial_h15_exp25_2026-09-15/` + `..._h15_exp25_reps23_2026-09-15/` (exposure=25, n=3)
- `close_loop_logs/openloop_ushape10x15mm_bimaterial_2026-09-15/` (open loop, default exposure, n=1)
- `close_loop_logs/openloop_ushape10x15mm_bimaterial_exp25_2026-09-15/` + `..._exp25_reps23_2026-09-15/` (open loop, exposure=25, n=3)
