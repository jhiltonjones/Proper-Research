# Control stack — v4

Everything from v3, plus the contact study: the experimental path that separates
"contact is a disturbance" from "contact is model".

```
analysis/      one analysis script per stack layer, plus the four-controller harness
controllers/   the four control laws and the two Jacobians, with provenance enforced
planning/      the patched L2 global smoother and the L3 time parameterisation
experiments/   the contact study — measurements, runner, analyser, tests
CONTROL_STACK.md   how the four layers fit together
```

## Where to drop each folder

`analysis/` and `experiments/` are standalone — they only need each other and
numpy (matplotlib optional, for figures). `controllers/` must sit somewhere the
project's `simulate_time_parameterized_*_mpc` modules are importable, because
the controllers wrap the real MPC classes rather than reimplementing them.
`planning/` replaces the corresponding project files; `planning/apply_patch.py`
re-applies the L2 upgrades to a fresh copy and `global_upgrades.diff` shows
exactly what changed.

## Run the tests first

None of these need the project package except `controllers/test_controllers.py`,
which says so clearly if you run it in the wrong place.

```bash
python experiments/test_contact_evidence.py     # 18 checks — the measurements
python experiments/run_contact_study.py --self-test   # 18 checks — the runner
python experiments/test_contact_study.py        # 18 checks — end to end
python planning/test_upgrades.py                # the L2 keep-out and tolerance work
python controllers/test_controllers.py          # needs the project MPC modules
```

Each suite tests in **both directions**. `test_contact_study.py` in particular
builds a synthetic study in which contact really is a pure additive offset, and
asserts the analyser declines to claim otherwise. A pipeline that can only
confirm is not evidence.

## What is new in v4

**Jacobian provenance is enforced, not documented.** Every controller records
where its Jacobian came from and whether contact was used in building it —
`controller.jacobian_is_contact_free`, `controller.jacobian_provenance`,
`controller.describe()`. `declare_provider()` refuses a bare callable outright,
because the one moment the truth is available is construction: a closed loop
cannot tell from its own output which Jacobian it was given, and both conditions
produce a plausible run.

**The contact study.** Three stages of increasing cost — offline Jacobian
divergence, the closed-loop factorial, the disturbance-observer ceiling sweep —
and four verdicts, each stated with what would have falsified it. See
`experiments/README.md`.

## The order of work

```bash
# 1. does the stack behave? (v3 analysis, unchanged)
python analysis/analyse_stack.py results/<run>

# 2. do the two Jacobians even differ? minutes.
python experiments/run_contact_study.py --stage offline --output-dir results/contact_study

# 3. if they do — the factorial, then the rebuttal
python experiments/run_contact_study.py --stage closed   --output-dir results/contact_study
python experiments/run_contact_study.py --stage observer --output-dir results/contact_study

# 4. four verdicts, four figures, one report
python experiments/analyse_contact_study.py results/contact_study
```

Every cell of the study writes the same 100-column record schema as the existing
MPC simulation, so any cell can be opened in the replay viewer without
conversion — which is what makes a surprising cell watchable rather than only
tabulated.

## Install OSQP before running any of this for real

Without it the MPCs fall back to SciPy SLSQP, which satisfies constraints only
to its `ftol`. The overshoot is 2×10⁻⁷ to 9×10⁻⁶ of the acceleration limit —
numerically fine — but `simulate_time_parameterized_beam_output_mpc` checks
applied commands against `input_constraint_tolerance = 1e-8` and will raise on a
solution that is not actually violating anything.
