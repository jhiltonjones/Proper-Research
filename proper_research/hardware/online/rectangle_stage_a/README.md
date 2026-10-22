# Stage-A execution-layer-C pipeline

Offline plan → live controller run (open-loop / INV-2DOF-trim / MPC-LTV) →
offline Q-ablation → comparison plots. Everything here was built and
validated live on the rectangle shape on 2026-09-17; it should work
unchanged on any other plan `run_time_parameterization.py` produces (the
scripts read the plan's own start pose, never a hardcoded one).

## Why this exists (read before running)

The morning of 2026-09-17 started from a working assumption that INV-vs-MPC
comparisons from earlier sessions were meaningful. They were not: the
execution seam in use then (`q_target = q_meas + clip(dt*u, ±0.006)`,
reset from the *measured* joints every tick) only realizes **~20%** of a
commanded relative joint increment per `servoJ` tick — confirmed by a
dedicated identification experiment (see "The A/B/C identification test"
below) — so every controller running through it was being judged on a
seam that ate 80% of its command, not on its own control law.

The fix is a different seam, "execution layer C", used by every script in
this package:

```
q_cmd_{k+1} = q_cmd_k + clip(dt*u_k, ±max_joint_step_rad)      (accumulator,
                                                                  NOT reset
                                                                  from q_meas,
                                                                  NOT anchored
                                                                  to q_ref+trim)
then stream q_cmd_k -> q_cmd_{k+1} as 5 linearly-interpolated
absolute servoJ targets at 50Hz instead of 1 target at 10Hz.
```

`PathFollowConfig.accumulator_seam = True` (the seam) +
`PathFollowConfig.servo_stream_hz = 50.0` (the interpolation) together are
what "layer C" means throughout this package and its docstrings.

## The A/B/C identification test

Three conditions, identical commanded joint trajectory (a benign single-
joint minimum-jerk step, 0.12 rad, robot-only, no camera/beam):

| condition | seam | rate | G (realization gain) | mean command error | max command error |
|---|---|---|---|---|---|
| A | `q_meas + Δq` (the old seam) | 10Hz | **0.199** | 67.9 mrad | 96.4 mrad |
| B | accumulator | 10Hz | **1.000** | 11.8 mrad | 36.2 mrad |
| C | accumulator | 50Hz | **1.000** | 2.7 mrad | 8.7 mrad |

`G = (q_final-q0)·(q_cmd_final-q0) / ‖q_cmd_final-q0‖²` — how much of the
commanded final displacement was actually realized. The seam alone (A→B)
is what fixes realization; the 50Hz interpolation (B→C) is what fixes
command-tracking *smoothness* on top of that (a further ~4x drop in mean/
max error). Both matter for a continuously-manipulated beam.

## Two structural bugs found and fixed along the way

Both live in `proper_research/hardware/online/state_stream.py`'s
`RobotJointStream` and `close_loop_path_follow.py`'s `main()` — not specific
to this package, but this package's runs are what surfaced them:

1. **`RobotJointStream` could silently freeze.** `rtde_receive.getActualQ()`
   was observed to keep returning the SAME cached packet indefinitely (no
   exception) after a stretch of heavy CPU-bound work in the same process
   (an LTV Jacobian schedule build) — 73% of polls were exact repeats in one
   measured case. `_poll_body()` now only refreshes a slot's freshness
   timestamp when the raw reading actually changed (a real robot's
   joints/pose are never bit-identical two polls in a row), so
   `max_age_s`-based staleness now actually fires instead of silently
   accepting a frozen feed as live. `RobotJointStream.reconnect_receive()`
   does the actual fix (a fresh receive-connection handshake; a passive
   wait does not clear it) — `close_loop_path_follow.main()` calls it
   automatically right after the solver build if the reader is stale at
   that point, plus a matching camera-freshness wait (the reconnect itself
   briefly pauses the main thread, which was separately observed to leave
   the camera's last frame stale enough to trip `stale_vision` on the very
   first tick).
2. **`AnalyticalBeamJacobianProvider` is frozen, `mpc_ltv_offline`'s default
   Jacobian source used it anyway.** `PathFollowConfig.jacobian_source`
   defaults to `"analytical_beam"`, which builds a provider whose
   `__call__` ignores its `state` argument entirely (confirmed directly in
   `close_loop_tip_control.py`: `return self._jac.copy()`). Calling that
   once per reference sample to build an "LTV" schedule gives N identical
   copies of one frozen Jacobian — `mpc_lti` wearing an `mpc_ltv_offline`
   label. `run_mpc_ltv.py` works around this by building the schedule from
   `beam_jacobian_providers.from_model_bundle` instead (genuinely state-
   dependent — condition number varied 37k→5.5k→45k across the rectangle,
   vs. a constant ~62.7k from the frozen provider) and injecting it via
   `close_loop_path_follow._SCHEDULE_OVERRIDE`.

## Live-validated numbers (rectangle 10×15mm, 2026-09-17)

| condition | E_RMS(active) | E_max(active) | E_servo_RMS | P(rate sat) |
|---|---|---|---|---|
| open-loop-C (healthy reader, insertion-init bug present) | 2.38mm | 4.78mm | 4.4mrad | 11.3% |
| open-loop-C (insertion fixed, reader froze that run) | 1.60mm* | 4.66mm* | n/a | n/a |
| INV-2DOF-trim, kp=1.0, kn=1.0 (untuned) | 1.98mm | 6.29mm | 31.1mrad | n/a |
| **INV-2DOF-trim, kp=0.6, kn=0 (tuned)** | **1.34mm** | **2.72mm** | 5.1mrad | 0.0% |
| MPC-LTV, genuine schedule, default Q/Qp/R/Rd/N | 1.87mm | 5.86mm | 17.0mrad | 23.3% |

\* tip-tracking numbers only (camera-derived, unaffected by the frozen
joint reader in that specific historical run); no servo-error number is
trustworthy for that run. **Nobody has re-run open-loop-C with both fixes
present at once — do that (`run_open_loop_c.py`, now fixed) before citing
a clean open-loop-C servo-error number.** Every number after this table's
first two rows postdates the `RobotJointStream` fix and is trustworthy.

**INV beats both open-loop and (untuned) MPC.** The kp/kn ablation that
found kp=0.6 is below; the Q-ablation that tried and failed to explain
MPC's saturation the same way is below that.

### The kn ablation (posture/null-space term)

Motivated by an offline windup diagnosis on the untuned (kp=1,kn=1) run:
pending servo error (`‖q_cmd-q_meas‖`) visibly escalated during the harder
transit/cornering sections in lockstep with a high `c_wind` correlation
(the feedback correction repeatedly pushing the command farther in the
direction the robot hadn't caught up to yet) — command-layer windup despite
no explicit integral term, because the accumulator seam makes ALL feedback
persistent, not just the nominal feedforward.

| kn | E_RMS(active) | E_servo_RMS | P(rate sat) | mean‖Δq_null‖ |
|---|---|---|---|---|
| 1.0 | 1.98mm | 31.1mrad | n/a | 0.0210 |
| 0.5 | 1.59mm | 15.3mrad | 35.3% | 0.0044 |
| 0.25 | 1.65mm | 14.7mrad | 29.3% | 0.0021 |
| 0.0 | 1.55mm | 11.8mrad | 20.5% | 0.0000 |

Clean monotonic signature: kn↓ ⟹ servo error↓, saturation↓, tracking↑. The
null-space insertion component (`|ΔL_null|`, a diagnostic-only 7D-projector
quantity, never applied to the actual command) was negligible throughout
(≤0.005mm, an order of magnitude below the advancer's own 0.166mm step
quantization) and insertion-command reversals didn't correlate with beam-
error spikes — H2 ("advancer oscillation") was checked and not supported;
H1 (posture loop too aggressive for the execution bandwidth) was.

### The kp ablation (task-tracking gain, kn=0 fixed)

Fixed the architecture: this was `inv_2dof_trim`'s own **feedback trim**
(recomputed every tick, non-integrating), not the persistent nominal
feedforward — see `inverse_jacobian_2dof_trim.py`'s module docstring for
why that distinction is the whole point of the controller.

| kp | E_RMS(active) | E_max(active) | P(rate sat) | R_task(mrad) |
|---|---|---|---|---|
| 0.0 (genuine open-loop, same 2DOF impl) | 1.785mm | 3.865mm | 0.0% | 0.00 |
| 0.3 | 1.498mm | 3.119mm | 0.0% | 1.52 |
| **0.6** | **1.342mm** | **2.717mm** | 0.0% | 3.42 |
| 1.0 | 1.552mm | 4.657mm | 20.5% | n/a |

A clean U-shape: too little feedback (kp=0) leaves real tracking error on
the table; too much (kp=1) saturates and roughens the command. kp=0.6 wins
both RMS and max, with zero saturation and zero beam-error spikes above
4mm — the frozen Stage-A development point.

### The MPC Q-ablation (does state/posture tracking cause the saturation?)

MPC showed nearly the SAME signature as untuned (kp=1) INV: 23.3% rate
saturation, 17.0mrad servo-error RMS, R_q=7.8mrad — so the natural next
question was whether MPC's state-tracking cost `Q` (joint-space, pulling
toward `ref_state` every horizon stage — the MPC analogue of INV's kn) was
the cause. Tested via **exact counterfactual QP re-solves** (not an
algebraic decomposition — once a QP is constrained, "task part" vs "Q part"
of a saturated solution isn't a unique split), replaying every logged tick
at the real measured state through freshly-built controllers with Q
modified:

| condition | P(u active) | P(Δu active) | R_u (mrad/s) | med. c_e |
|---|---|---|---|---|
| baseline (Q=1.0) | 0.626 | 0.985 | 63.4 | 0.782 |
| Q≈0 (weight×0.001) | 0.737 | 0.992 | 80.5 | 0.761 |
| Q_no_null (row-space only + 1% residual) | 0.737 | 0.996 | 86.1 | 0.808 |

**Both ablations made saturation and roughness WORSE, not better** —
falsifying the hypothesis. Two structural findings came out of building
this: literal `Q=0` is DARE-infeasible for this system (closed-loop
spectral radius 1.000136, just barely unstable — nothing regulates the
null-space configuration, so the terminal cost-to-go has a marginal mode),
and so is pure `Q_N=0` (null-space-only removal). Q is load-bearing for the
terminal cost to be well-posed here, unlike INV's `kn` (a one-step
resolved-rate gain with no such requirement) — the kn↔Q analogy breaks
down structurally at exactly this point. The practical near-zero
substitutes used above (weight×0.001, +1% residual null cost) are the
smallest stabilizing values found by a direct search.

**Conclusion, not yet acted on:** the aggressive MPC command is not coming
from state/posture tracking. Most likely Qp (task-tracking demand) combined
with the horizon dynamics and the hard rate limits. The next step, if
picked back up, is NOT another Q sweep — see `q_ablation.py`'s module
docstring for the reasoning and what it would take to test that instead.

## Prerequisites

- A plan directory from `run_time_parameterization.py` (offline planner) --
  the `time_parameterized_configuration_path` subdirectory specifically,
  containing a `.npz` with `state`/`state_reference` and
  `desired_position_m`.
- Robot on External/Remote Control per your site's setup, camera connected.
  Every script here calls `common.preflight()` first (camera health check,
  reset to the plan's own start pose, robot safety check) -- do not add
  `--skip-preflight` unless you already just ran one against the SAME
  plan and know the robot hasn't moved since.
- `OSQP` installed for the MPC scripts (falls back to slower/less exact
  SciPy SLSQP otherwise -- see `proper_research/controllers/README.md`'s
  "Install OSQP" section).

## Running

**One-shot, everything:**

```bash
python -m proper_research.hardware.online.rectangle_stage_a.run_all \
    --plan-dir plans/rectangle_10x15mm_bimaterial/time_parameterized_configuration_path \
    --out-dir close_loop_logs/stage_a_$(date +%Y%m%d) \
    --schedule-cache /tmp/rectangle_genuine_ltv_schedule.npy \
    --run-q-ablation
```

First run builds the genuine LTV schedule (~90-150s, one real beam solve
per reference sample) and caches it at `--schedule-cache`; repeat runs
against the same plan load it instantly.

**Stage by stage** (useful for re-running just the one that aborted, or
iterating on gains without re-running everything):

```bash
# 1. acceptance test -- run this FIRST on any new plan/shape
python -m proper_research.hardware.online.rectangle_stage_a.run_open_loop_c \
    --plan-dir <plan> --out-dir <out>

# 2. tuned INV (kp=0.6, kn=0 is the frozen development point; sweep with --kp/--kn)
python -m proper_research.hardware.online.rectangle_stage_a.run_inv_2dof_trim \
    --plan-dir <plan> --out-dir <out> --kp 0.6 --kn 0.0

# 3. MPC-LTV, genuine schedule (slow the first time, see above)
python -m proper_research.hardware.online.rectangle_stage_a.run_mpc_ltv \
    --plan-dir <plan> --out-dir <out> --schedule-cache <cache.npy>

# 4. compare + plot (no robot)
python -m proper_research.hardware.online.rectangle_stage_a.plot_comparison \
    --run <out>/runs/openloopff_accumC_<ts> \
    --run <out>/runs/inv_2dof_trim_kp0.6_kn0_<ts> \
    --run <out>/runs/mpc_ltv_accumC_<ts> \
    --labels open-loop-C,INV-2DOF-trim,MPC-LTV \
    --out-dir <out>/comparison_plots

# 5. (optional, no robot) does state/posture tracking cause MPC's saturation?
python -m proper_research.hardware.online.rectangle_stage_a.q_ablation \
    --plan-dir <plan> --mpc-run-dir <out>/runs/mpc_ltv_accumC_<ts> \
    --schedule-cache <cache.npy> --out <out>/q_ablation.npz
```

`plot_comparison.py` prints the standard metrics table (`E_RMS(active)`,
`E_max(active)`, `E_servo_RMS/max`, `P(rate sat)`, `R_q`, `P(err>4mm)`) and,
with `--out-dir`, writes `error_vs_time.png`, `servo_error_vs_time.png`
(only for accumulator-seam runs), and `tip_trajectory.png`.

## Files

| file | what it is |
|---|---|
| `common.py` | plan-state loading, pre-flight reset, robot/camera health checks -- shared by every run script |
| `run_open_loop_c.py` | `u=u_ref` exactly, execution layer C, no correction -- run first on any new plan |
| `run_inv_2dof_trim.py` | tuned INV (default kp=0.6, kn=0), execution layer C |
| `run_mpc_ltv.py` | `mpc_ltv_offline`, default Q/Qp/R/Rd/N, genuine per-sample-relinearised schedule, execution layer C |
| `q_ablation.py` | offline (no robot): does removing MPC's state/posture cost fix its saturation? (no -- see above) |
| `plot_comparison.py` | offline (no robot): metrics table + overlay plots across any set of runs |
| `run_all.py` | chains the above as subprocesses |

The controller itself lives in
`proper_research/controllers/inverse_jacobian_2dof_trim.py` (read its module
docstring for the full architecture), wired into
`proper_research/hardware/online/controller_adapters.py`'s
`build_offline_solver` as `controller_kind="inv_2dof_trim"` -- no
monkeypatching needed to use it outside this package either.
`controller_kind="open_loop_ff"` (`TrivialFeedforwardController`, also in
`controller_adapters.py`) is the same story.

## What "execution layer C" requires of a `PathFollowConfig`

Every run script here sets these; if you're building a new one by hand,
don't forget any of them:

```python
cfg.feedforward_joint_trajectory = False   # NOT anchored to q_ref+trim
cfg.accumulator_seam = True                # the seam itself
cfg.servo_stream_hz = 50.0                 # the 50Hz interpolation
cfg.joint_velocity_limit_rad_s = 0.10      # >= the plan's own peak |u_ref|
cfg.max_joint_step_rad = 0.010             # == dt * joint_velocity_limit_rad_s (KEEP CONSISTENT)
cfg.initial_insertion_m = <read from the plan, NOT the 0.030 dataclass default>
```

`max_joint_step_rad == dt * joint_velocity_limit_rad_s` matters beyond
cosmetics for `inv_2dof_trim` specifically: that controller reproduces the
harness's own accumulator update internally (`q_cmd6` mirror) to compute
its returned velocity; if the two bounds are inconsistent the harness's own
velocity-then-step clip re-clips more tightly than the controller's
position-space clip intended, and the mirror drifts out of lockstep with
the harness's real `q_cmd`.
