# `flow/` — Layer 5: flow-only centreline following

Two parts:

1. **The feasibility study** (`STEP0_AUDIT.md` → `analyse_layer5_flow.py`) — a
   cheap observability analysis of a four-element circumferential flow ring,
   testing the predictions in `../FLOW_CENTERING_THEORY.md` and stopping. Nothing
   in it imports `controllers/` or the beam model.
2. **The flow-fed closed loop** (`CLOSED_LOOP.md`) — added afterwards at the
   user's request, past the theory document's hard stop: a DARE / Riccati
   inverse-Jacobian controller tracking the `bends_p30_p0` trajectory with the
   tip error estimated from the simulated ring instead of vision. This imports
   the planner context and the beam plant but does **not** modify
   `controllers/`.

Start with `STEP0_AUDIT.md` — in particular gate 0.1 (there is no flow-sensor
data path in the codebase; this layer is green-field).

## Modules

| file | role | self-test |
|---|---|---|
| `flow_observability_reference.py` | **the oracle** — `numpy` + `scipy.special` only, no project imports. Reproduces every table in the theory document (§4.2, §4.3, §4.4, §5.2, §6). Ground truth for everything else. | `python -m proper_research.flow.flow_observability_reference --self-test` |
| `flow_field.py` | STEP 1 — Womersley + steady flow, harmonic-series waveform, `grad_phasor` / `vmean_phasor` / `Gamma`, Dean correction as a separate switchable term (T1–T3) | `... flow_field --self-test` |
| `sensor_model.py` | STEP 2.1–2.2 — exact 4-channel ring readings, named channel combinations, `quadrupole_second_order()` | `... sensor_model --self-test` |
| `flow_jacobian.py` | STEP 2.3 — `M = d(measurement)/d(offset)` by central differences; `FlowMeasurementProvider` with provenance (mirrors `beam_jacobian_providers`); `compose_with_beam_jacobian(M, P, J)` | `... flow_jacobian --self-test` |
| `pipelines.py` | STEP 2.4 — `scalar_aggregate` vs `differential` providers over one model; `compare_sensor_pipelines()` (the D1 gate) | `... pipelines --self-test` |
| `../analysis/analyse_layer5_flow.py` | STEP 3–5 — the six-point geometry cross, D1–D6, the validity pass, and the `report / metrics / PNG` triple | `... analysis.analyse_layer5_flow --self-test` |

### Closed loop (past the theory document's hard stop — see `CLOSED_LOOP.md`)

| file | role | self-test |
|---|---|---|
| `riccati_inverse_jacobian.py` | DARE / Riccati feedback gain with integral action, built on `controllers/inverse_jacobian_controller.py` (untouched) | `... flow.riccati_inverse_jacobian --self-test` |
| `flow_feedback.py` | `FlowTipEstimator` — simulated ring → tip-position estimate for the controller | `... flow.flow_feedback --self-test` |
| `simulate_flow_closed_loop.py` | closed-loop runner: `planning_context` + the `bends_p30_p0` reference + contact plant, flow feedback vs vision vs open loop | `... flow.simulate_flow_closed_loop --self-test` |

Run the study:

```
python -m proper_research.analysis.analyse_layer5_flow \
    --output-dir proper_research/analysis/results/layer5_flow \
    --sigma-over-vmean 0.02
```

## Headline results

* **D1** — the scalar-aggregate map is rank 1 and singular at the setpoint
  (`4|e|/R`); the differential map is rank 2 and best-conditioned there
  (`8a/R`). Reproduced to ~1e-7. This is the justification for retaining the
  four raw channels.
* **D2** — `Gamma/Gamma_P` collapses with the Womersley number (knee at α ≈ 4–5,
  two orders down by α ≈ 15); flat along the bend-angle arm of the cross to
  numerical precision. Reproduces the §4.2 table to 5e-4.
* **D3** — with a physiological steady component, observability is retained all
  cycle; a low-gain window (≈ 90 ms/beat here) opens only for a low-steady,
  high-PI trace. The gain null is offset from the flow null by the §4.4 phase
  separation.
* **D4** — Dean bias grows with De, outward, strongest on the sharp bend, and
  **exceeds the 0.10 R tolerance** there. *Partial disagreement:* the bias is
  **not** flat along the radius arm (De couples to R through Re as well as to the
  bend), so the cross does not mirror D2's isolation.
* **D5** — the quadrupole is exactly zero for a parabolic profile, non-zero under
  Womersley (peak near α ≈ 5), and its residual is a Dean detector only for
  `|e|/R ≲ 0.05`.
* **D6** — the feasibility boundary sits at **R ≈ 5.8–6.6 mm** for a 7 Fr
  catheter at 1–2 % noise, set by α. Reproduces the §6 boundary radii exactly.
  Sensor quality (~2 mm per 5× noise) and catheter calibre (~1 mm per doubling)
  are weak levers. **The limit is the flow profile, not the instrument.**

Anatomy: comfortable in coronary/radial/peripheral vessels; usable in
carotid/femoral (bend correction mandatory); marginal at the iliac; infeasible
in the aorta.

## Assumptions (none of this is in the codebase — see `STEP0_AUDIT.md`)

* blood: ρ = 1060 kg/m³, μ = 3.5e-3 Pa·s, 72 bpm
* `sigma_s` unknown → every noise-driven number is reported as `sigma_s / v_mean`
* bent-segment arclength 30 mm (feeds `R_c = L_bend / angle`)
* physiological waveforms are synthetic but documented; PI and harmonic content
  are reported
* catheter matched to calibre: 3 Fr (a = 0.55 mm) below 2.5 mm, 7 Fr (1.15 mm)
  otherwise
