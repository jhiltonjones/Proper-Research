# Controllers

Four control laws and the two Jacobians they are compared under. The
implementations live here rather than in the comparison harness deliberately:
any controller detail living in the harness is a place the harness could favour
one rung over another.

| file | what it is |
|---|---|
| `beam_jacobian_providers.py` | Contact-aware and contact-free 3×7 tip Jacobians, with provenance attached. |
| `inverse_jacobian_controller.py` | Damped resolved-rate control with nullspace configuration regulation. |
| `mpc_variants.py` | The three MPC rungs: LTI, LTV offline schedule, LTV SQP online. |
| `test_controllers.py` | All four against the real MPC modules, on a mock plant. |

All four expose the same call:

```python
step = controller.solve(
    measured_state=z,            # (7,) [q1..q6, insertion]
    measured_beam_position=p,    # (3,) measured tip
    control_index=k,
    previous_input=v_prev,       # (7,)
)
```

and return a `ConfigurationMPCStep`-shaped object, so they drop into
`simulate_time_parameterized_beam_output_mpc`'s loop and into
`compare_controllers.py` unchanged.

## Is the Jacobian contact-free? Ask the controller.

You no longer have to remember. Every controller records where its Jacobian
came from, and the record travels with the run:

```python
controller.jacobian_is_contact_free   # True / False
controller.jacobian_provenance        # the full dict
controller.describe()["jacobian"]     # one printable line
```

```
CONTACT-FREE | name=no_contact source=bundle.models['no_contact'] contact_enabled=False lumen_query=False
```

This is enforced rather than encouraged. `declare_provider()` **refuses a bare
callable**:

```python
>>> build_inverse_jacobian_controller(jacobian_provider=lambda z: my_jac(z), ...)
TypeError: This Jacobian source has no provenance, so the run could not record
whether it is contact-free. Pass a BeamJacobianProvider (see from_model_bundle /
contact_and_contact_free), or set allow_undeclared=True if this is a mock in a
unit test.
```

The reason is narrow and worth stating. The single most expensive mistake
available in this comparison is running what you believe is the contact-free
condition against a Jacobian that quietly had contact in it — or the reverse —
and discovering it after the numbers are in a paper. A closed loop cannot detect
this from its own output: both conditions produce a plausible run. The only
moment the truth is available is construction, so that is where it is captured,
and a source that cannot say what it is does not get to be used by default.

`allow_undeclared_jacobian=True` exists for unit tests with mock Jacobians and
is threaded through every builder. Do not set it in a study.

## The two Jacobians

`build_model_bundle` already constructs both models and hands them back
together — `models = {"plant", "contact", "no_contact"}`. The `contact` model
has `ContactConfig(enabled=True, use_in_jacobian=True)` and a `LumenQuery`; the
`no_contact` model has `ContactConfig.disabled()` and no lumen query. They are
separate instances with separate equilibrium caches, so differentiating one
cannot perturb the other. That is why `jacobian_model` is a separate argument to
`controller_factory_joint_space.build_controller` in the first place.

There are three routes to a contact-free Jacobian and they are **not the same
experiment**:

**1. A separate model instance** — `from_model_bundle(contact=False)`.
Structurally contact-free: the contact energy is never assembled, never
differentiated, and the equilibrium the derivative is taken at is itself
contact-free. This is "the controller does not know the beam touches anything".

**2. The flag on a shared model** — `contact_free_jacobian_scope(model)`. Sets
`model.contact_cfg.use_in_jacobian = False` and restores it afterwards. The
forward solve still includes contact (`energy_from_u` uses `use_contact`) while
the sensitivity drops it (`energy_gradient_u` is called with
`use_contact=use_contact_in_jacobian` inside `implicit_tip_jacobian`). So the
derivative is taken **at the contact equilibrium** but **of the contact-free
energy**. Often the more realistic condition: the plant really is in contact,
and only the linearisation is wrong.

**3. Whatever the pack holds** — `from_controller_pack()`. Convenient, but what
you get depends on how the pack was built, so check `describe()`.

For the four-controller comparison, use 1 or 2 and apply it **identically to all
four**. Every provider carries a `describe()` naming the model and its contact
settings, and the harness prints it, so a mismatch shows up in the report rather
than in a conclusion.

```python
from beam_jacobian_providers import contact_and_contact_free, compare_providers

contact, contact_free = contact_and_contact_free(
    bundle=bundle, controller_pack=controller_pack
)
print(contact_free.summary_line())
print(compare_providers(contact, contact_free, reference.state))
```

That last call is worth running before the comparison. If the two Jacobians
barely differ, the model mismatch you are asking the controllers to reject is
not there, and four near-identical controllers would mean nothing.

## The ladder

Each rung differs from the one below it in exactly one thing, which is what
makes the comparison informative.

| controller | model | what the rung buys |
|---|---|---|
| `naive_inverse_jacobian` | `J` at the measured state, inverted | — the baseline |
| `mpc_lti` | one `J`, frozen at reference sample 0 | preview and constraint handling |
| `mpc_ltv_offline` | one `J` per reference sample, precomputed | scheduling the model along the path |
| `mpc_ltv_sqp_online` | `J` at the measured state, every step | relinearising where the plant actually is |

**naive → LTI** prices preview and constraints. If the naive controller is
close, the horizon is not buying much on this trajectory.

**LTI → LTV offline** prices scheduling. A large gap means the Jacobian varies
materially over the run.

**LTV offline → SQP online** prices relinearising on the measured state. The
offline schedule is evaluated at `z_ref`; if the plant has drifted away from the
reference — which is exactly what a contact-free Jacobian against a contact
plant provokes — then `J(z_ref)` is the derivative at the wrong point.

You can predict that last gap before running anything: `analyse_layer4_mpc.py`
computes the affine model's one-step prediction error against what the plant
actually did. On your existing run it is 47.5 µm rms against 43.6 µm of per-step
tip motion — a ratio of 1.09. An affine model whose error exceeds the motion it
predicts is not a good local model, so the online rung should win. Be suspicious
if it does not.

Read accuracy and worst-case solve time together. Online relinearisation rebuilds
`Jbar`, the Hessian and the OSQP `P` values every step; an accuracy win you
cannot compute inside one sample period is not a win.

## Two details that are not decoration

**The nullspace term in the resolved-rate controller is mandatory.** The
beam-position Jacobian is 3×7, so four directions of joint motion leave the tip
where it is. Without a nullspace term the configuration drifts along them until
it hits a limit or a singularity, and the comparison measures that drift rather
than the control law. It pulls toward the reference configuration — the same
trajectory the MPCs track in their state cost.

**`nominal_reference_positions_m` is worth passing deliberately.** It defaults to
`reference.desired_position_m`, the centreline — but the planned configuration
does not put the tip exactly on the centreline, so the planner's own tracking
error is absorbed into `d_hat` along with genuine plant mismatch and the two stop
being separable. Passing the planner's *achieved* tip positions makes `d_hat`
mean what its name says. All three MPC builders take it.

## Running

```bash
python test_controllers.py                     # no project imports
python compare_controllers.py --self-test      # the harness
python compare_controllers.py --output-dir results/<run>/controller_comparison
python compare_controllers.py --jacobian-model flag    # the other experiment
python compare_controllers.py --jacobian-model contact # the control condition
```

`--lti-freeze-index` chooses where the LTI rung linearises. 0 is the honest
default: the start of the trajectory is the only point a genuinely
time-invariant design could know in advance. Freezing at the midpoint flatters
the LTI rung and should be labelled if used.

`--sqp-inner-iterations N` above 1 repeats the solve, re-evaluating the Jacobian
at the first predicted state — a short SQP rather than a full one. Each extra
iteration costs a Jacobian evaluation and a full QP solve, and shows up directly
in the worst-case solve time.

`--sqp-relinearise-horizon` keeps the offline schedule's *shape* and shifts it by
the difference between the measured-state Jacobian and the schedule's value at
the current index, instead of writing one model into every horizon slot. Neither
costs an extra beam solve.

## Install OSQP before running this for real

Without OSQP the MPCs fall back to SciPy SLSQP, which satisfies constraints only
to its `ftol`. `test_controllers.py` measures the resulting overshoot at
2×10⁻⁷ to 9×10⁻⁶ of the acceleration limit — numerically fine, but
`simulate_time_parameterized_beam_output_mpc` checks applied commands against
`input_constraint_tolerance = 1e-8` and will raise
`Applied command violates a hard velocity/acceleration bound` on a solution that
is not actually violating anything. SLSQP also fails its line search
occasionally on the first solve, before any warm start exists. The test reports
both as backend warnings rather than controller failures, and says so.
