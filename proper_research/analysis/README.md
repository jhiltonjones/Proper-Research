# Control-stack analysis

Six scripts. Five read files your pipeline already writes and never import the
robot or beam model, so they run anywhere. The sixth runs controllers and needs
the project.

| file | reads | answers |
|---|---|---|
| `analyse_layer1_inverse.py` | `inverse_configuration_path.csv` + summary | did the inverse solve stay far inside tolerance, and was the continuation healthy? |
| `analyse_layer2_global.py` | `global_configuration_path.csv`, dense CSV, summary (+ optional inverse dir) | did smoothing converge, what did it spend the tolerance budget on, and did any L1 constraint disappear? |
| `analyse_layer3_time_parameterization.py` | the timed trajectory CSV | which axis binds, and did smoothing actually buy traversal time? |
| `analyse_layer4_mpc.py` | `configuration_mpc_simulation.csv` + summary | how much of the closed-loop error was already in the plan, is the affine beam model good enough, would it meet its deadline? |
| `analyse_stack.py` | all of the above | one error budget across all four layers |
| `compare_controllers.py` | the timed trajectory + your project | four controllers, one plant, one reference |

Each writes `<stem>_report.md` (readable), `<stem>_metrics.json` (machine
readable) and a PNG. Matplotlib is optional — without it the numbers still come
out.

## Install

Drop the folder in, e.g., `proper_research/analysis/`. Every script runs both as
a module and as a plain file next to its siblings. Only `numpy` is required.

## The one idea worth knowing

Layers 1 and 2 optimise the same quantity with opposite semantics.

* **L1 minimises** tip error — it is a residual in a least-squares objective, so
  the solver drives it toward zero and typically spends a small fraction of the
  tolerance.
* **L2 constrains** tip error — it appears only as `‖p(χ) − p*‖ ≤ tol`, which is
  free until it binds. Every micron of tip error inside the ball is a micron of
  joint smoothness bought for nothing.

So the tip error is *expected* to grow across L2, up to the tolerance and no
further. Growth beyond the tolerance is a constraint violation, not a trade. The
"budget" column in every report is `error / tolerance`, which is what makes
layers with different tolerances comparable.

## Layer analysis

```bash
python analyse_stack.py \
  --inverse-dir   results/<run>/offline_inverse_configuration_60 \
  --global-dir    results/<run>/global_configuration_full_debug \
  --reference-dir results/<run>/time_parameterized_configuration_path \
  --mpc-dir       results/<run>/time_parameterized_beam_output_mpc_simulation \
  --output-dir    results/<run>/stack_analysis
```

Any layer may be omitted; missing ones are reported, not fatal.

Two optional arguments earn their keep:

* `analyse_layer3 --baseline-dir <timed-from-unsmoothed-path>` turns the
  derivative proxies into an actual traversal-time claim. Nothing else in the
  stack can make that claim honestly.
* `analyse_layer4 --reference-dir <timed-path>` splits the measured tip error
  into the part the plan already had and the part the controller added.

`analyse_layer3` sniffs column names (several spellings per quantity), so it
survives schema drift in the time-parameterisation writer. If it guesses wrong
the report names the column it used — check that line first.

## Controller comparison

```bash
python compare_controllers.py --self-test          # no project imports needed
python compare_controllers.py --output-dir results/<run>/controller_comparison
```

Four controllers, in increasing order of machinery:

1. `naive_inverse_jacobian` — damped resolved-rate, nullspace-regulated. The
   nullspace term is not decoration: the beam-position Jacobian is 3×7, so four
   directions of joint motion leave the tip alone and would drift without it.
2. `mpc_lti` — your QP with one Jacobian frozen at the first reference sample.
3. `mpc_ltv_offline` — your current controller, unchanged.
4. `mpc_ltv_sqp_online` — relinearised at the *measured* state each step;
   `--sqp-inner-iterations N` re-linearises along the prediction.

The ladder prices each rung: preview and constraints (1→2), scheduling the model
along the path (2→3), relinearising on the measured state (3→4).

### What keeps it fair

* One `run_closed_loop` function steps the plant for all four, so the beam
  commit sequence is identical and cannot favour anyone.
* Same reference, plant, initial state, sample period, actuator limits,
  tolerances, terminal hold.
* **Jacobians are built contact-free; the plant keeps contact on.** A
  context manager forces `adapter.model.contact_cfg.use_in_jacobian = False` for
  the whole comparison and restores it afterwards. Every controller therefore
  rejects the same declared model mismatch. If the toggle is missing the run
  aborts with an explanation rather than quietly comparing something else; pass
  `--allow-contact-jacobian` to override, and the report says so.
* Worst-case solve time is reported next to accuracy. A controller that wins on
  error and misses its deadline has not won.

Each controller writes `<output-dir>/<name>/configuration_mpc_simulation.csv` in
**exactly** your existing 100-column schema — verified column-for-column, in
order. So `analyse_layer4_mpc.py` runs on each one unchanged, and so does any
viewer you already point at that file.

### What it needs from the project

Resolved at import, with a named error if absent:

```
simulate_time_parameterized_configuration_mpc     load_configuration_reference,
                                                  make_default_mpc_config,
                                                  _beam_snapshot, _magnet_pose
simulate_time_parameterized_beam_output_mpc       BeamOutputTrackingMPC,
                                                  BeamOutputMPCConfig,
                                                  resolve_analytical_beam_jacobian,
                                                  precompute_reference_beam_jacobians
proper_research.planning.planning_context         build_planning_context
controller_pack                                   plant_diagnostic_joint_adapter,
                                                  forward6d_plant
```

The two private base helpers (`_beam_snapshot`, `_magnet_pose`) are the only
private coupling; both are called in one place each, near the bottom of `main`.

`--self-test` exercises the loop, the record schema and the report against mock
objects, so you can check the harness before wiring the project in.

## Reading a report

Findings are graded `PASS` / `WARN` / `FAIL` / `NOTE` and repeated at the top.
`FAIL` means a stated invariant is broken — a violated tolerance, a dropped
constraint, an optimiser that stopped at its iteration cap. Fix those before
reading anything else in the report: an unconverged solve is not evidence about
a formulation.
