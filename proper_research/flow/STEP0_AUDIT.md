# STEP 0 — audit (Layer 5, flow-only centreline following)

Nothing was built in this step. Findings, gate status, and the build estimate
follow. The rest of the layer proceeds only because gate 0.1 is satisfied *in
substance* — see 0.1.

> **Scope note.** This audit and the `analyse_layer5_flow` deliverable stop at
> the feasibility map, as `flow_prompt.txt` requires. A closed-loop controller
> (`riccati_inverse_jacobian.py` + `flow_feedback.py` +
> `simulate_flow_closed_loop.py`, see `CLOSED_LOOP.md`) was added **afterwards at
> the user's explicit request**, past that hard stop. It does not modify
> `controllers/`.

---

## 0.1 — THE CRITICAL ONE: where do the four channels collapse to a scalar?

**Finding: there is no flow-sensor data path anywhere in the codebase.**

Traced end to end:

| looked for | result |
|---|---|
| a driver / HAL for a flow sensor | none. `hardware/` is UR-robot IK/Jacobian validation and magnet kinematics only |
| a sim source of flow channels | none. `grep -ri "flow"` over `*.py` hits only `charactisations.py` (a variable name) and `sensitivity.py` (unrelated). No `womersley`, `poiseuille`, `dean`, `flow_field`, `flow_sensor`, `circumferential` anywhere |
| a log / replay / CSV column carrying raw channels | none |
| the 4→1 aggregation function | **does not exist — there are no four channels to aggregate** |
| sample rate / units / per-channel filtering | not applicable |
| channel-to-`phi_k` mapping | not applicable |

The only exteroceptive modality in the tree is **vision** (`vision/`: `camera.py`,
`detect_blue.py`, `vision_w_tangnet.py`, `line_fit_through_points.py`, …), which
is camera-based and produces a tip pose / centreline-deviation estimate in pixels.

**Gate status: PASS (in substance), with the reason stated.**

The gate exists to stop the project if raw per-channel data is being *destroyed*
at an aggregation point that cannot be changed without a firmware/hardware change.
That situation does not exist here: there is no sensor, no firmware, no
aggregation point, and therefore nothing being discarded. Layer 5 is entirely
green-field.

The gate's second clause — *"do not work around it by simulating channels and
pretending"* — is respected: STEP 1–3 are an explicitly-labelled analytical
feasibility study. The measurement model (`sensor_model.py`) retains all four
channels **by construction** and never claims to be reading hardware. No number
in the deliverable is presented as a hardware measurement; every noise-driven
figure is parameterised by `sigma_s / v_mean` because `sigma_s` is unknown
(0.7).

**Consequence for a future hardware/firmware spec (out of scope here, recorded
so it is not lost):** when a real ring is fitted, the driver and logger MUST
persist the four raw channels (with the `phi_k` map and per-channel timestamps)
and MUST NOT aggregate to a scalar anywhere upstream of the estimator. The D1
result below is the reason.

---

## 0.2 — Lumen geometry and R_c

`simulation/magnetic_beam/lumen_geometry.py` and
`simulation/simulations/scenario.py` both define `LumenConfig` / `LumenBend`:

* **radius** — constant `LumenConfig.radius` (default 4 mm in `scenario.py`,
  10 mm in `lumen_geometry.py`), OR an arclength-indexed
  `radius_profile = ((s, r), …)` linearly interpolated and held flat outside its
  range. So `R` is a function of arclength when a profile is supplied, constant
  otherwise.
* **bend** — `LumenBend(bend_angle_rad, bend_start, bend_end, bend_axis)`. The
  centreline tangent is rotated by `bend_angle_rad` across `[bend_start,
  bend_end]` with a `smoothstep` profile (`make_lumen_centerline_multi_turning`).

**R_c derivation.** A `LumenBend` turns the tangent through `theta =
bend_angle_rad` over an arclength `L_bend = bend_end - bend_start`. A circular arc
of that length and turn angle has

```
R_c = L_bend / theta
```

The Dean number needs `L_bend`, which the study does not get to observe, so
`analyse_layer5_flow.py` takes `L_bend = 30 mm` as a **flagged assumption**
(`BEND_ARCLENGTH_M`) and reports `R_c`, `De` and the Dean bias against it.
Implemented as `flow_observability_reference.radius_of_curvature_from_bend`.

**Historical bends actually used.** Run directories and filenames on this branch
show `bends_p0_p20` / `bends_0_20` (0°, +20°) and a deleted `csv_data_bend_100`
(100°). `scenario.py` default radius 4 mm; `run_contact_study.py` scales the
radius *down* toward neurovascular calibre and scales bend angles for a
tortuosity sweep. The six-point cross in the analysis uses R ∈ {1.5, 3.5,
5.5} mm and bend ∈ {0, 20, 100}°, consistent with that history.

---

## 0.3 — Centreline and tip pose

* **Planned centreline after L1–L3**: stored as CSV
  (`inverse_configuration_path.csv`, `global_configuration_path.csv`,
  `time_parameterized_configuration_path.csv`) and consumed output-only by the
  `analysis/` scripts. The geometric centreline `C` (N×3) and radius `R` (N) are
  built by `lumen_geometry.build_lumen_geometry` and wrapped as
  `contact.LumenQuery(C, R)`.
* **Accessor**: `LumenQuery.closest(p) -> (distance, R_local, q)` gives the
  nearest centreline point `q` and interpolated local radius; there is **no
  stored local frame / tangent**. As the theory doc anticipates, the transverse
  projection `P` must be built from the tangent — done in
  `flow_jacobian.transverse_projection(tangent)`, which returns a 2×3 matrix
  whose rows span the plane ⟂ to the tangent. The tangent itself is a finite
  difference of consecutive `C` points.
* **Measured/simulated tip position**: from the beam forward model
  (`sensitivity_optimized` / `forward_model_optimized`), tip `p` and tip tangent
  available as `solution.tip` and via `implicit_tip_jacobian(...).tip_tangent_base`.

The transverse offset the estimator wants is `e = P (p_tip - q)`.

---

## 0.4 — Jacobian provider conventions

`controllers/beam_jacobian_providers.py`:

* **Interface** — `BeamJacobianProvider(name, jacobian_at, adapter, model,
  source)`; `__call__(state)` returns the 3×7 tip-position Jacobian; `describe()`
  returns a provenance dict (`name`, `source`, contact settings, `model_id`);
  `summary_line()` one-liner.
* **Provenance** — `declare_provider(provider, allow_undeclared=False)` splits a
  source into `(callable, provenance)` and **raises on a bare callable** unless
  `allow_undeclared` (mocks only). `provenance_line()` renders it.
* **`contact_and_contact_free(*, bundle, controller_pack, jacobian_mode)`** —
  returns `(contact_provider, contact_free_provider)` over one model bundle,
  sharing robot kinematics and differing only in the beam.
* **`compare_providers(first, second, states)`** — Frobenius / relative
  difference (max, rms) over a set of states, with both `describe()` blocks and a
  `note`.
* Differencing: `SensitivityOptions.difference_scheme ∈ {"forward","central"}`,
  `eps_theta = 1e-6`, all seven RHS solved in one factorisation.

**Mirrored by the flow pipelines:**

| beam | flow |
|---|---|
| `BeamJacobianProvider` | `flow_jacobian.FlowMeasurementProvider` (name, `measurement_at`, `jacobian_at`, `describe()`, `summary_line()`) |
| `contact_and_contact_free()` | `pipelines.sensor_pipelines(ring)` → `(scalar_aggregate, differential)` |
| `compare_providers()` | `pipelines.compare_sensor_pipelines(scalar, differential, offsets)` |
| central differences, `eps` | `flow_jacobian.central_difference_jacobian`, `DEFAULT_EPS = 1e-6`, `DIFFERENCE_SCHEME = "central"` |

---

## 0.5 — Analysis conventions

`analysis/_stack_common.py` + `analyse_layer4_mpc.py`:

* **Triple** — `Report.write(dir, stem)` emits `<stem>_report.md` and
  `<stem>_metrics.json`; a PNG via `new_figure` / `save_figure` with the house
  `PALETTE`. `analyse_layer5_flow.py` writes `layer5_flow_report.md`,
  `layer5_flow_metrics.json`, `layer5_flow_feasibility.png`.
* **Grading** — `Finding(level ∈ {"ok","warn","fail","info"})` rendered as
  `PASS / WARN / FAIL / NOTE`; `Report.worst_level` aggregates.
* **Budget** — `budget_report(name, errors, tolerance)` with
  `utilisation = error / tolerance`; `budget = 1` sits on the constraint. Layer 5
  uses `delta_e / R` against the `0.10` tolerance throughout.
* **`--self-test`** — `compare_controllers.py --self-test` runs the harness
  against mock objects with no project imports. `analyse_layer5_flow.py
  --self-test` does the same: a local ~15-line Womersley snippet
  (`numpy` + `scipy.special` only) checks the D1/D2/D6 identities without loading
  `flow/` or the beam model.

---

## 0.6 — Fluid parameters

**Finding: none exist.** No density, viscosity, flow rate, heart rate, or
waveform anywhere in the tree (`grep` over `parameters.py` and the whole
package). Blood defaults are used and **flagged as assumptions** in every output:

```
rho = 1060 kg/m^3        mu = 3.5e-3 Pa.s        72 bpm  ->  omega = 7.540 rad/s
```

(`flow_field.RHO_BLOOD / MU_BLOOD / OMEGA_BLOOD`, echoed into
`layer5_flow_metrics.json["assumptions"]["fluid"]`).

Representative axial speeds per vessel (radial 0.10, coronary 0.15, carotid 0.30,
femoral 0.25, iliac 0.30, aorta 0.40 m/s) are literature values, also flagged.

---

## 0.7 — Sensor characterisation

**Finding: nothing recorded** — no modality (thermal / Doppler / pressure), no
transfer function, no noise level, no bandwidth, no calibration.

**Gate status (0.7): `sigma_s` is unknown, so every feasibility number is
reported parameterised by `sigma_s / v_mean`, never as a single figure.** The
headline run uses `sigma_s / v_mean = 2%` as a *stated assumption*; D3 and D6
sweep it (0.5, 1, 2, 5%). This is enforced structurally:
`resolvable_offset_fraction(alpha, a/R, sigma_over_vmean, dean_bias)` takes the
ratio, not `sigma_s`.

---

## What exists vs what was built

| item | status | est. |
|---|---|---|
| flow-sensor driver / log / raw-channel retention | **absent; not built** (feasibility study needs no hardware) | — |
| `LumenConfig` / `LumenBend`, centreline CSVs, `LumenQuery` | exists, reused unchanged | — |
| beam Jacobian + provider pattern | exists, reused unchanged (composition helper only) | — |
| `_stack_common` report/metrics/PNG/budget | exists, reused unchanged | — |
| oracle `flow_observability_reference.py` | **built** — the theory doc referenced it as existing; it did not. `numpy` + `scipy.special` only, reproduces every FLOW_CENTERING_THEORY.md table | ~4 h |
| `flow/flow_field.py` (Womersley + steady + waveform; T1–T3) | **built** | ~3 h |
| `flow/sensor_model.py` (exact 4-channel; combos; quad 2nd-order) | **built** | ~3 h |
| `flow/flow_jacobian.py` (M by central diff; provenance) | **built** | ~2 h |
| `flow/pipelines.py` (scalar vs differential; D1 gate) | **built** | ~2 h |
| `analysis/analyse_layer5_flow.py` (D1–D6, STEP 4, triple, `--self-test`) | **built** | ~6 h |

## Gate failures

None that halt the work. Two are flagged and handled rather than blocking:

* **0.1** — no raw-channel data path exists. Not a data-destruction failure; the
  layer is green-field and the analytical study retains all four channels by
  construction. Recorded requirement for a future hardware spec.
* **0.7** — `sigma_s` unknown. Every feasibility number is parameterised by
  `sigma_s / v_mean`.

## Deviations from the prompt's assumptions

* The prompt treats `flow_observability_reference.py` and the companion docs
  `CONTROLLER_THEORY.md` / `EXPERIMENT_DESIGN.md` as existing. Only
  `FLOW_CENTERING_THEORY.md` and `CONTROL_STACK.md` exist. The oracle was
  rebuilt from the theory document's numbers and its self-tests reproduce every
  table in it (§4.2, §4.3, §4.4, §5.2, §6) to the quoted precision.
