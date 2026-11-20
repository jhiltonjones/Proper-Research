# The four-layer control stack

```
L1  offline inverse configuration   chi(s), node by node        task error is a COST
L2  global constrained smoothing    a smoother chi(s)           task error is a CONSTRAINT
L3  time parameterisation           chi(t) under actuator limits
L4  beam-output MPC                 velocity commands, beam feedback
```

Each layer hands the next a strictly poorer guarantee than you might assume, and
almost every confusing result in this stack comes from that. Read this section
before the reference.

## The one idea

**Layers 1 and 2 optimise the same quantity with opposite semantics.**

L1 puts tip error in a least-squares residual, scaled by the tolerance:

```
r(chi) = [ (p - p*)/tol_p ; (t - t*)/(2 sin(tol_t/2)) ; continuity ; centring ]
minimise r^T r
```

so the solver drives tip error toward zero and typically spends a small
fraction of the tolerance.

L2 discards that residual. Its objective is first and second differences of the
scaled configuration plus a seed pull and a joint-centring term — all functions
of `chi` alone, with **no task-space term at all**. Tip position reappears only
as a hard inequality at each node:

```
|| p(chi_j) - p*(s_j) || <= tol_p
```

A constraint is free until it binds. Every micron of tip error inside the ball
is a micron of joint smoothness bought for nothing, so the optimiser spends the
whole budget and the tip migrates to the surface of the ball, independently at
each node. **Tip error is expected to grow across L2, up to the tolerance and
no further.** Growth beyond the tolerance is a constraint violation, not a
trade — and the layer-2 analysis tells the two apart.

Three more properties follow from the formulation and surprise people:

* **L2 constrains nodes only.** Nothing is imposed between them; dense
  validation only *checks*. Node feasibility does not imply a feasible path.
* **L2's beam baselines are frozen per refinement round.** Within a round the
  beam map is a deterministic function of `chi`, warm-started from a
  path-dependent equilibrium captured once from that round's seed. Hysteresis
  is frozen, not removed.
* **L1 is history-dependent.** Each accepted node is committed into the beam
  solver before the next is solved. A controller replaying those configurations
  sees the same tip only if it reproduces the same commit sequence.

## Tolerances are not the same size at every layer

| layer | position tolerance | typical spend |
|---|---|---|
| L1 | 10 µm | ~0.3× (it minimises) |
| L2 | 10 µm | up to 1.0× (it only bounds) |
| L4 | 1500 µm | ~0.2× |

A micron of planner roughness is invisible against a millimetre-scale
controller tolerance. Every report in the analysis package prints
`error / tolerance` — the **budget** — precisely so layers with different
tolerances can be compared without fooling yourself.

---

# Running the stack

## L1 — inverse configuration

```bash
python run_inverse_head_exclusion.py \
    --source-magnet-lumen-exclusion-radius-m 0.12 \
    --require-analytical-magnet-jacobian
```

Writes `offline_inverse_configuration_60/`. The exclusion radius is not
optional in practice: without it the planner has no keep-out and the magnet may
sit inside the head. With it, expect the minimum margin to be *exactly zero* —
the constraint is active and shaping the whole path.

```bash
python analyse_layer1_inverse.py --inverse-dir results/<run>/offline_inverse_configuration_60
```

Look for: task error far below tolerance, few multistarts, no step halving, no
rank-deficient Jacobians, keep-out margin ≥ 0.

## L2 — global smoothing

**Use `run_global_configuration.py`, not `debug_global_from_saved_inverse.py --full`.**
The debug driver's defaults are diagnostic: 20 iterations and zero refinement
rounds. A run that stops at its iteration cap is an intermediate trust-region
iterate; interior iterates of a trust-region method routinely look rough, and
nothing downstream should be judged from one.

```bash
python run_global_configuration.py \
    --exclusion-radius 0.12 \
    --auto-tolerance \
    --balance-smoothing-weights \
    --tolerance-spend-fraction 0.5
```

What each flag does:

`--exclusion-radius` restores the source-magnet / lumen keep-out. L2 previously
had no exclusion constraint of any kind — the fields were deserialised from the
inverse result and passed through, never re-imposed and never re-checked. This
is the only genuine safety regression in the stack, and it is the first thing to
turn on. Pass the same radius L1 used.

`--auto-tolerance` sets the global tolerance from what L1 actually achieved
rather than a number it never met. This matters more than it looks: a *single*
node outside tolerance makes the seed infeasible, and
`preserve_feasible_seed_on_failure` retains the untouched seed only when the
seed's own violation is inside `constraint_tolerance`. One bad node therefore
disables the safety net for the entire path, and feasibility restoration runs
instead — with a slack-weighted objective that is allowed to move the tip. The
driver's preflight reports this before the solve rather than after.

`--balance-smoothing-weights` sets the per-coordinate second-difference weights
so every coordinate contributes equally to the objective at the seed. Smoothing
under a fixed quadratic objective is a trade: the coordinate whose term
contributes least is the cheapest to sacrifice. With a uniform weight that is
insertion — the axis with a 25× tighter acceleration limit than the joints and
a first-derivative demand ten times larger — so a pass that improves every joint
can make the binding axis worse.

`--tolerance-spend-fraction` bounds how much of the tip tolerance smoothing may
consume. The optimiser is constrained against `fraction × tolerance` while
still being *reported* against the physical tolerance, so the remainder becomes
margin the controller inherits instead of budget the smoother spends. One
number, no new mathematics; this is the cheapest fix for a wandering tip.

`--tip-centring-weight` is the alternative: pay for a centred tip rather than
forbid an off-centre one. It adds

```
0.5 * w * sum_j || (p_j(chi_j) - p*_j) / tol_p ||^2
```

linearised at the round's seed, `p_j(chi_j) ~ p_j^0 + J_j (chi_j - chi_j^0)`.
That is one extra block-diagonal quadratic in the same QP — the problem stays
convex, the Hessian stays sparse, and trust-constr needs no new machinery.
Linearising at the seed is consistent with the rest of the layer, which already
freezes its beam baselines per round.

Use the fraction *or* the weight, not usually both. The fraction is a hard
guarantee with no tuning; the weight is a soft preference that keeps the whole
tolerance available where the geometry needs it.

```bash
python analyse_layer2_global.py \
    --global-dir results/<run>/global_configuration_converged \
    --inverse-dir results/<run>/offline_inverse_configuration_60
```

Look for, in order: did it converge, does it violate its constraints, did any L1
constraint disappear, where did the tolerance budget go, which coordinate paid
for the smoothing.

## L3 — time parameterisation

Two speed profiles over the same geometric path:

```bash
python run_time_parameterization.py --profile optimal    # the default
python run_time_parameterization.py --profile standard   # trapezoidal baseline
python run_time_parameterization.py --profile both       # and compare
```

**optimal** is what the module already did: backward controllable sets at every
grid point, then forward at the maximum admissible path acceleration. It is
time-optimal for the given path and limits.

**standard** is the textbook trapezoidal schedule most path-following
implementations use — one cruise speed that respects every actuator everywhere,
one ramp acceleration that also respects every actuator everywhere, ramp in,
cruise, ramp out. Feasible by construction and conservative by construction: a
single worst-case speed for the whole path rather than a speed that follows the
local limit.

Only `s_dot(s)` differs between them. Same caps, same grid, same beam
validation, same downstream code — so a duration difference is attributable to
the profile and nothing else. `--profile both` writes the two to sibling
directories and saves `profile_comparison.json` with the difference.

This is also how you answer "did the smoothing help?" honestly. Derivative
proxies are not a substitute:

```bash
python run_time_parameterization.py --profile optimal --global-dir <smoothed>  --output-dir <a>
python run_time_parameterization.py --profile optimal --global-dir <unsmoothed> --output-dir <b>
python analyse_layer3_time_parameterization.py --reference-dir <a> --baseline-dir <b>
```

Look for: which axis binds, and whether limits are active at all. A time-optimal
profile is bang-bang — some actuator should be at a limit almost everywhere. Low
saturation means something else sets the duration and a smoother path will not
make it faster.

## L4 — beam-output MPC

```bash
python simulate_time_parameterized_beam_output_mpc.py
python analyse_layer4_mpc.py --mpc-dir results/<run>/..._mpc_simulation \
                             --reference-dir results/<run>/time_parameterized_configuration_path
```

Look for: whose error it is (the `--reference-dir` split), whether the affine
beam model predicts well one step ahead, and whether every QP fits inside one
sample period.

## The whole stack at once

```bash
python analyse_stack.py \
    --inverse-dir   results/<run>/offline_inverse_configuration_60 \
    --global-dir    results/<run>/global_configuration_converged \
    --reference-dir results/<run>/time_parameterized_configuration_path \
    --mpc-dir       results/<run>/..._mpc_simulation \
    --output-dir    results/<run>/stack_analysis
```

One error budget across all four layers, with each layer's tip error against its
own tolerance.

## Comparing controllers

```bash
python compare_controllers.py --self-test          # no project imports
python compare_controllers.py --output-dir results/<run>/controller_comparison
```

Four controllers — resolved-rate inverse Jacobian, LTI MPC, LTV offline-Jacobian
MPC, SQP online-Jacobian MPC — through one closed-loop function against one
plant, with Jacobians built contact-free while the plant keeps contact on. Each
writes `configuration_mpc_simulation.csv` in the existing 100-column schema, so
`analyse_layer4_mpc.py` runs on each unchanged.

---

# File reference

## New files

| file | what it is |
|---|---|
| `global_upgrades.py` | The keep-out constraint (imports L1's own `_MagnetLumenExclusionConstraint`, does not re-derive it), the tip-centring quadratic, the tolerance-spend helper, and the weight balancer. |
| `time_parameterization_profiles.py` | The standard trapezoidal profile and a context manager that swaps it in for the optimal one. |
| `run_global_configuration.py` | Production driver for L2: seed preflight, convergence settings, weight balancing, the new safety knobs. |
| `run_time_parameterization.py` | L3 driver with `--profile optimal/standard/both`. |
| `analysis/` | Six analysis scripts; see `analysis/README.md`. |

## Changed file

`global_constrained_configuration_path.py` gains seven config fields and the
hooks that use them. Every default reproduces the previous behaviour exactly —
`source_magnet_lumen_exclusion_radius_m=None`, `tolerance_spend_fraction=1.0`,
`tip_centring_weight=0.0` — and the module's own `--self-test` passes unchanged.
`global_upgrades.diff` is the reviewable diff.

The keep-out is threaded through **five** places, and all five matter:

1. the hard solve's constraint list;
2. the restoration solve's constraint list — restoration may relax the *task*
   constraints through their slacks, never the keep-out, because recovering
   tracking feasibility by moving the magnet into the head would be a worse
   outcome than failing to recover it;
3. `_recomputed_constraint_violation`, so candidate selection sees it;
4. **the seed-feasibility gate.** This one is easy to miss. Without it a seed
   that is tracking-feasible but sits inside the exclusion region reads as
   violation-free, and `preserve_feasible_seed_on_failure` would retain that
   unsafe path in preference to the optimiser's safe one — silently turning the
   safety net into a safety hazard;
5. the saved summary, as `source_magnet_lumen_exclusion`.

## Verification

* `python global_constrained_configuration_path.py --self-test` — the module's
  own suite, unchanged, on the patched file.
* `python test_upgrades.py` — the three upgrades on a mock adapter: a keep-out
  that is *binding* at the seed (margin −4.81 mm) is satisfied after smoothing
  (+0.61 mm); spend fractions of 1.0 / 0.5 / 0.25 bound the realised budget to
  1.000 / 0.070 / 0.002; the tip cost reduces tip rms from 175 µm to 120 µm.
  The test asserts the keep-out is binding at the baseline, so it cannot pass
  vacuously.
* `python compare_controllers.py --self-test` — the controller harness and its
  record schema against mock objects.

## Installation

```
proper_research/planning/
    global_upgrades.py                       (new)
    global_constrained_configuration_path.py (patched)
    time_parameterization_profiles.py        (new)
    run_global_configuration.py              (new)
    run_time_parameterization.py             (new)
proper_research/analysis/                    (new package)
```

`global_upgrades.py` and `time_parameterization_profiles.py` both carry the same
dual import the rest of the planning package uses, so they work as a package or
as loose files for review.

---

# Things worth knowing

**`globally_feasible` can never be True with dense validation disabled.** It is
computed as candidate feasibility AND node feasibility AND dense feasibility,
and the dense term is False whenever validation is off — regardless of solution
quality.

**`dense_validation_samples_per_interval` writes a header-less file when there
are no samples.** If dense validation is disabled,
`global_configuration_dense_validation.csv` is created completely empty. The
layer-2 analysis handles this; your own readers may not.

**Refinement inserts one node per round.** `maximum_refinement_rounds=0` means
the loop runs once and terminates with `maximum_refinement_rounds_reached` — a
message that sounds like a considered stop and is really "the single permitted
round ended".

**The MPC's residual estimator conflates two errors.**
`nominal_reference_positions_m` defaults to `reference.desired_position_m`, the
centreline — but the planned configuration does not put the tip on the
centreline. The planner's own tracking error is therefore absorbed into `d_hat`
along with genuine plant mismatch, and the two are no longer separable. Passing
the planner's *achieved* tip positions instead would make `d_hat` mean what its
name says. At a 1.5 mm tolerance this is small; it will not be if you tighten.

**The tangent tolerance never binds.** At 80° allowance against ~32° realised
error, tangent is inactive everywhere. Anything that says the global layer
"trades tracking accuracy for smoothness" should say *position* accuracy.
