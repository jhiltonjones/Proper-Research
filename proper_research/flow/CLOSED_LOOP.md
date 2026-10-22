# Flow-fed closed loop — DARE inverse-Jacobian controller

Beyond the Layer-5 feasibility study (`STEP0_AUDIT.md`, `FLOW_CENTERING_THEORY.md`):
a closed-loop simulation that tracks the pre-computed L1→L2→L3 trajectory with
the tip error **estimated from the simulated flow ring instead of vision**.

This goes past the theory document's hard stop deliberately, at the user's
request; it uses the D3 coast-time and D4 bias findings as its design inputs
(the integral term below is the direct answer to "the Dean bias is a bias, not
noise — it does not average out").

## Pieces

| file | role |
|---|---|
| `riccati_inverse_jacobian.py` | the controller: resolved-rate inverse Jacobian with a **DARE feedback gain** and integral action, built on `controllers/inverse_jacobian_controller.py` (untouched) |
| `flow_feedback.py` | `FlowTipEstimator` — true tip → nearest centreline → transverse offset → simulated 4-channel ring (noise, cardiac phase) → gain-compensated inversion → tip-position estimate |
| `simulate_flow_closed_loop.py` | the runner: `planning_context.build_planning_context` + the saved `bends_p30_p0` reference + the contact beam plant, four control conditions through one loop |

## The DARE / Riccati formulation

Kinematic integrator `z_{k+1} = z_k + dt·u_k`, redundant 3×7 tip Jacobian `J`,
`P` the 2×3 projection onto the plane ⟂ the centreline tangent (basis anchored
to world `+z`, so flip-free), `J_t = P·J`.

The flow ring measures the transverse tip error `y_k = P(p*_k − p_tip(z_k))`.
A commanded **transverse tip velocity** `v_k` moves it as
`y_{k+1} = y_k − dt·v_k`. Augment with the integral of `y` and solve a DARE for
the Riccati-optimal PI gain on the tip error:

```
x = [ y ; ξ ]        ξ_{k+1} = ξ_k + dt·y_k

A = [[ I₂   , 0  ]]   B = [[ −dt·I₂ ]]   Q = diag(Q_y, Q_ξ)   R = R_v
    [[ dt·I₂, I₂ ]]       [[ 0      ]]

P = AᵀPA − AᵀPB(R + BᵀPB)⁻¹BᵀPA + Q
K = (R + BᵀPB)⁻¹BᵀPA          # 2×4, computed once
v = −K·[y ; ξ]
```

The tip velocity is realised through the **existing beam Jacobian**, exactly as
the resolved-rate baseline maps a task velocity to joints:

```
w = J_t⁺·v  +  (I − J_t⁺J_t)·k_n·(z_ref − z)/dt       # J_t⁺ lightly damped;
u = u_ref + w                                          # projector undamped/exact
```

* **Integral action** rejects the plant-model mismatch and the Dean-flow bias
  (both biases — a proportional law leaves a steady-state offset). The integral
  is accumulated as a **world-frame 3-vector** and re-projected into the current
  transverse frame each step, so the frame rotating through the bend does not
  corrupt the accumulated correction (a real failure mode: a frame-local
  integral drifts the tip back out by ~sin(bend angle) after the initial
  correction).
* **`K` is Jacobian-independent** — `A`, `B` contain no `J_t`, so the Riccati
  gain is a fixed optimal PI matrix. This is itself a result: for tip-error
  regulation the Riccati gain does not depend on the beam model; what the model
  buys is `J_t⁺`, the mapping to joints.
* The nullspace projector uses the **undamped** `J_t⁺`, so it is an *exact*
  rank-5 projector and never fights the tip correction. The four tip-invariant
  joint directions are regulated toward the reference configuration.
* **Insertion is locked out** of the feedback (`insertion_locked`): the advance
  axis follows the feedforward schedule and is never recruited for centring —
  otherwise the ring's transverse correction fights the along-track plan (this
  was a real failure mode during development).
* Clipping (velocity → acceleration → state box) is identical to the baseline.
* The controller **never receives the true tip** in the `flow` run — only the
  ring's reconstruction `p_meas = p* + Pᵀ·ê`.

## The flow estimator

Per step: nearest centreline point + local tangent → true transverse offset
`e_true`; build the local Womersley field (`α` from `R_local`) and a
`SensorRing`; read four channels at the current cardiac phase; add sensor noise
`σ_s` (referenced to the cycle-mean flow via `σ_s/v_mean`); invert the two
differential channels with the known instantaneous gain `g(φ) = 2u'(a,φ)` plus a
few Gauss–Newton steps against the exact readout. When `|g|` falls into the
low-gain window (D3) the estimate is **held**. A hard deadband suppresses
sub-noise wobble.

## Running it

```
# feasibility study first (unchanged)
python -m proper_research.analysis.analyse_layer5_flow

# closed loop — four conditions through one plant
python -m proper_research.flow.simulate_flow_closed_loop \
    --initial-joint-offset-deg 3 -2 1 0 0 0 \
    --sigma-over-vmean 0.02 --max-steps 751
```

Conditions:

| run | feedback | what it shows |
|---|---|---|
| `riccati_flow` | simulated flow ring | the deliverable |
| `riccati_vision` | true tip (perfect) | upper bound — the cost of losing the camera |
| `no_dare_flow` | flow, resolved-rate only | what the DARE adds |
| `feedforward_only` | none | the uncorrected plant-model mismatch |

## Tuning note (real plant)

On the mock plant (`--self-test`, `J_model = J_plant`, straight lumen) a
near-deadbeat DARE gain converges in ~0.2 s. On the **real** plant it does not,
and two separate mechanisms had to be tamed:

1. **Near-deadbeat proportional gain oscillates.** The contact plant differs from
   the contact-free control Jacobian and the transverse frame rotates through
   the 30° bend, so the effective loop gain is not the identity the deadbeat
   gain assumes → a fast oscillation. Fix: a modest `tip_tracking_weight /
   tip_velocity_weight` ratio (`3e2 / 8e0`), leaving gain margin.
2. **A strong integral goes lightly-damped-complex** against the same mismatch —
   a ~2 s limit cycle where the tip corrects to <0.05 mm, drifts back out, and
   repeats. Fix: keep the integral *very* small (`integral_weight 2e0`) — enough
   to trim a slow bias, not enough to set up the cycle — and accumulate it in a
   **world-frame** 3-vector so the rotating frame does not corrupt it.

3. **Any lag in the flow estimate destabilises the loop.** With the thin phase
   margin left after (1) and (2), even a 2-step measurement smoothing tips the
   loop back into a limit cycle. Fix: feed the estimate **promptly and
   unsmoothed** (`average_window 1`, `ema_tau_s 0`) — a prompt noisy estimate
   tracks better than a smooth lagged one, and the small deadband handles the
   noise near the setpoint.

With the shipped defaults (`tip_tracking_weight 3e2`, `tip_velocity_weight 8e0`,
`integral_weight 2e0`, `nullspace_gain 0.1`; prompt unsmoothed ring estimate)
the real plant corrects the 0.67 mm initial offset and tracks the trajectory
well inside the 1.5 mm tolerance — see `flow_closed_loop_report.md` for the
numbers against the perfect-vision, no-DARE, and open-loop baselines. This is
the honest lesson: a Riccati gain designed against the nominal model needs
detuning for the model it will actually meet — the D4 point that a bias needs
margin, not speed.

Each writes `configuration_mpc_simulation.csv` in the existing schema (so
`analysis/analyse_layer4_mpc.py` runs on it), with extra `flow_*` columns, plus
`flow_closed_loop_report.md` / `_metrics.json` / `.png`.

## Self-tests (no project sim imports, mock plant + straight lumen)

```
python -m proper_research.flow.riccati_inverse_jacobian --self-test
python -m proper_research.flow.flow_feedback --self-test
python -m proper_research.flow.simulate_flow_closed_loop --self-test
```

## Assumptions on top of the feasibility study

* `σ_s` uncharacterised — every result reported at a stated `σ_s/v_mean`;
* ring radius `a = 0.9 mm` inside a ~1 mm catheter;
* physiological waveform is the documented steady-dominant synthetic trace;
* the plant is the contact beam model; the controller Jacobian is contact-free,
  so the flow-fed integral has a real mismatch to reject.
