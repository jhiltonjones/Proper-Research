# The contact study

Four files that answer one question:

> In a highly-contacted anatomy, is contact a **disturbance** the controller's
> estimator can absorb, or is it **model** — something that changes the
> input-output sensitivity itself, so that no estimator can substitute for
> putting it in the Jacobian?

| file | what it is |
|---|---|
| `contact_evidence.py` | The measurements. No simulation, no I/O — just the four quantities the question turns on. |
| `run_contact_study.py` | The experiment. Three stages of increasing cost. |
| `analyse_contact_study.py` | Four verdicts, each stated with what would have falsified it. |
| `test_contact_evidence.py`, `test_contact_study.py` | Both directions: the pipeline must find the effect when it is there, and decline to claim when it is not. |

```bash
python test_contact_evidence.py                       # 18 checks, no project imports
python run_contact_study.py --self-test               # 18 checks, mocks only
python test_contact_study.py                          # 18 checks, end to end
python run_contact_study.py --stage offline --output-dir results/contact_study
python run_contact_study.py --stage closed  --output-dir results/contact_study
python run_contact_study.py --stage observer --output-dir results/contact_study
python analyse_contact_study.py results/contact_study
```

## The algebra the whole thing rests on

A controller linearises the plant as `p ≈ p_nom + J (z − z_ref) + d`.

If contact were a **disturbance**, it would enter only through `d`: an additive,
input-independent offset. `J` would be unchanged and the beam-output MPC's
existing estimator would absorb it. Once converged, the contact-free controller
would match the contact-aware one.

If contact is **structural**, it changes `J`. Then

```
d_hat = (J_contact − J_free)(z − z_ref)
```

and `d_hat` is not a disturbance at all — it is a linear function of *the
controller's own tracking error*, with the Jacobian gap as its coefficient
matrix. The estimator is chasing a target its own actions create. No filter
setting fixes that, and the study's fourth stage is there to prove it by trying.

That identity is why the study is worth running rather than argued: it is an
equation with two measurable sides.

## The three stages

**Stage 1 — `offline`.** No plant stepping, no controller. The two Jacobians
evaluated at the same states. Minutes. **Run it first**: if the two agree to a
fraction of a percent everywhere, there is nothing to find and the expensive
stages would only measure solver noise.

Two numbers, and they must be read together:

- **relative gap** `‖ΔJ‖/‖J‖` — how much the sensitivity changes.
- **largest principal angle** between the row spaces — whether the *set of joint
  motions that reach the tip* changes. A row space is a 3-dimensional subspace
  of joint space; scaling a row, or mixing rows that already span it, leaves it
  untouched. Only a genuine change in which directions reach the tip rotates
  it, and **no additive output disturbance can rotate a sensitivity**.

A large gap with no rotation is a *gain error* — still not something an
estimator can correct, but a weaker claim, and the analyser reports it as the
weaker claim rather than rounding it up.

The angles come back in **ascending** order. Element 0 is the smallest and is
near zero whenever the two subspaces share any direction, which they nearly
always do; the number that says "these are different subspaces" is the last one.

**Stage 2 — `closed`.** The factorial: anatomy × Jacobian model × controller ×
initial condition. Every cell writes the same 100-column CSV the existing
layer-4 analysis and the replay viewer already read, so any cell can be opened
in the viewer without conversion.

Per cell it also fits the regression above and writes `contact_diagnosis.json`.
Two details there are not cosmetic:

- The regression uses the **instantaneous** residual, not the filtered one. With
  `disturbance_filter_alpha > 0` the recorded `d_hat` is lagged, and lag alone
  would depress R² for reasons unrelated to the hypothesis.
- The Jacobian gap is measured **on the states the run actually visited**. If
  the run drifted, the relevant gap is the gap where it drifted to.

**Stage 3 — `observer`.** The rebuttal. On the tightest anatomy, with the
contact-free Jacobian, sweep `disturbance_filter_alpha` across its admissible
range, and include the contact-aware cell at α = 0 as the ceiling. Without that
ceiling line, a flat α sweep is just a flat line with nothing to be flat
*relative to*. If the contact-free penalty survives the best α, "you tuned the
observer badly" is no longer available as an explanation.

## Why the anatomy sweep scales the radius, not the tortuosity

This is the design decision that makes the dose–response interpretable, and it
is worth being stubborn about.

Scaling the **radius** with the centreline held fixed leaves the desired tip
path unchanged, so **one planned trajectory stays valid across the whole
sweep**. One reference, one set of limits, one initial condition; the only thing
that differs is how hard the beam is pressed. That is a controlled dose.

Scaling the **tortuosity** changes the centreline, so the plan changes, so the
configurations differ, so the Jacobians differ, so the joint-limit margins
differ — and a penalty that grew across such a sweep could be blamed on any of
them. `--tortuosity-scales` exists because reviewers ask for it, but each level
needs its own re-planned reference (`--anatomy-json`), the runner refuses to
reuse the baseline plan on a changed centreline, and those cells are marked
`confounded=1` so no figure quietly mixes them with the controlled arm.

## The controls

| held fixed | why |
|---|---|
| The plant is contact-enabled in **every** cell | The Jacobian factor changes what the controller *believes*, never what happens. `_assert_plant_contact` refuses to run otherwise — a "contact matters" result obtained by also disabling contact in the plant would be an artefact. |
| One `ConfigurationMPCConfig` per (anatomy, Jacobian) pair | A difference in limits between rungs would masquerade as a difference in control law. |
| Seeded initial conditions, index 0 = the planned start exactly | The same physical initial condition reaches every controller and every Jacobian model. Random per-cell offsets would measure luck. |
| Initial condition is the outermost loop | A study stopped early still leaves every (anatomy, Jacobian, controller) combination represented, rather than a lopsided subset. |
| One `run_closed_loop` | The plant is stepped and committed identically regardless of which law produced the command. |

## The four verdicts

`analyse_contact_study.py` answers four questions and, for each, says what
would have falsified it:

| | claim | falsified by |
|---|---|---|
| **V1** | the sensitivity itself changes | the two Jacobians agreeing everywhere, with no row-space rotation |
| **V2** | what the observer calls a disturbance is the missing model term | `d_hat` poorly explained by the tracking error — or well explained, but by a matrix unlike the measured gap |
| **V3** | the penalty grows with contact | a flat penalty in engagement, bootstrap slope interval containing zero |
| **V4** | no observer setting rescues it | some α bringing the contact-free controller within 20% of the contact-aware ceiling |

V2 separates its two failure modes on purpose. "Explained but not recovered"
means something input-dependent is driving the residual but contact is not
established as the whole of it — and the first thing to check is whether
`nominal_reference_positions_m` was passed, because the planner's own tip error
lands in `d_hat` otherwise and the two stop being separable.

The dose axis prefers **measured** engagement (contact energy or contacting-node
fraction, read from the beam's own diagnostics) over the radius knob, and the
report says which was used. Radius is a setting; engagement is what the beam
experienced, and only the second is comparable across anatomies.

## Reading the result honestly

The verdicts can come back negative, and `test_contact_study.py` exists mainly
to prove they will: it builds a synthetic study where contact is a pure additive
offset and checks the analyser reports `none` / `not shown` / `flat` / `observer
closes the gap`. Any pipeline can confirm. The test of an evidence pipeline is
whether it will tell you no.

If V1 comes back `none`, the useful conclusion is not "the method failed" — it
is that at this radius the beam is barely touching, and the sweep needs to reach
tighter calibre before the question has content.
