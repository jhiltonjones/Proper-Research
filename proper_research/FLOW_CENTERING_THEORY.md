# Flow-only centreline following — theory, feasibility, and experimental plan

**Layer 5 of the control stack.** Replace the vision estimate of tip-to-centreline
deviation with an estimate derived from the four-element circumferential flow
sensor array, and characterise where that substitution is admissible.

Companion to `CONTROLLER_THEORY.md` (rungs A1–A4) and `EXPERIMENT_DESIGN.md`
(gaps 1–3). Everything below is *estimation* theory; the control ladder is
unchanged and is reused as-is.

Scope decisions taken for this document: four-sensor ring, pulsatile (Womersley)
flow, **feasibility as the headline claim**, and a hard stop after the
no-closed-loop observability analysis. No estimator and no controller are
specified here beyond what is needed to state the feasibility criterion.

---

## 0. The one-sentence result

> Flow-only centring is limited by the **Womersley number**, not by sensor noise.
> Below α ≈ 4 the pulsatile profile keeps a usable radial gradient and centring is
> comfortable; above α ≈ 10 the core moves as a plug, the radial gradient inside
> the core vanishes, and no amount of sensor quality recovers it. At 72 bpm this
> puts the boundary at a vessel radius of roughly **5–6 mm**.

And a second result that is free:

> The array already carries the direction information. Collapsing the four
> channels to one scalar destroys it — the scalar map is **rank 1 and singular at
> the setpoint**, the differential map is **rank 2 and best-conditioned at the
> setpoint**. This is a data-path change, not a hardware change.

---

## 1. Problem statement

The catheter advances along a lumen whose centreline is already planned by
layers L1–L3. Vision previously supplied the tip deviation from that centreline.
It is now withdrawn. The only exteroceptive signal is a ring of four flow sensors
at the tip.

Let

* `R(z)` — vessel radius at arclength `z` along the centreline
* `a` — catheter radius, so the sensors sit on a circle of radius `a` about the catheter axis
* `e = (e_y, e_z)` — transverse offset of the catheter axis from the vessel centreline
* `u(r, t)` — axial flow speed at radius `r` from the **vessel** centreline
* `φ_k = kπ/2`, `k = 0..3` — sensor angular positions, `u_k = (cos φ_k, sin φ_k)`

The control objective is `e → 0` while `ż > 0`. The estimation objective is to
recover `e` from the four readings `s_k`.

---

## 2. The measurement model

### 2.1 Exact readings

Sensor `k` sits at vessel-frame radius `|e + a u_k|`, so

```
s_k = u(|e + a u_k|, t) + n_k,      n_k ~ N(0, σ_s²)
```

with `|e + a u_k|² = |e|² + 2a (e·u_k) + a²`.

### 2.2 Small-offset expansion

For `|e| ≪ R`, writing `u' = ∂u/∂r` evaluated at `r = a`:

```
s_k ≈ u(a) + u'(a) (e·u_k) + O(|e|²)
```

The four channels then decompose into four orthogonal combinations, and each one
means something different:

| combination | value | carries |
|---|---|---|
| `s₀ − s₂` | `2 u'(a) · e_y` | **lateral offset, y** |
| `s₁ − s₃` | `2 u'(a) · e_z` | **lateral offset, z** |
| `¼ Σ s_k` | `u(a) + O(|e|²)` | mean flow → `v_mean`, and via `Q = πR²v_mean`, the **local radius** |
| `s₀ − s₁ + s₂ − s₃` | `0` to first order (exactly zero only for a parabolic profile — see §5.2) | **axisymmetry residual** — a free diagnostic |

Four sensors, four independent pieces of information. The two differential
channels are the estimate; the common mode gives longitudinal context; the
quadrupole channel is identically zero under the model and therefore detects
*violations* of it — Dean secondary flow, catheter tilt, or a failed sensor.

### 2.3 Why the differential channels are the whole game

Both differential channels are **linear in `e`** and their gain `2u'(a)` **does
not depend on `e`**. In particular it does not vanish at `e = 0`. This is the
opposite of the scalar case below, and it is the single structural fact that
makes the project feasible.

---

## 3. Observability

### 3.1 Rank structure

Define the measurement map `M = ∂s/∂e`.

**Scalar aggregate (current pipeline).** The linear terms cancel in the mean, so
the leading dependence is quadratic:

```
s̄ = v_max (1 − (|e|² + a²)/R²)     ⟹     ∂s̄/∂e = −2 v_max e / R²
```

Two consequences, both fatal:

1. The gradient is **parallel to `e`**. Only the radial direction is sensed; the
   azimuthal direction is unobservable by rotational symmetry. `rank M = 1`.
2. The gradient **vanishes at `e = 0`** — the measurement Jacobian loses rank
   exactly at the setpoint being regulated to. `rank M = 0` at the target.

Normalised by `v_mean/R`, the scalar sensitivity is `4|e|/R`.

**Differential (proposed).** From §2.2, normalised the same way, the sensitivity
is `8a/R` in *both* transverse directions. `rank M = 2` everywhere, and the
conditioning is independent of `e`.

| \|e\|/R | scalar `4\|e\|/R` | differential `8a/R` | ratio | scalar rank |
|---|---|---|---|---|
| 0.00 | 0.000 | 1.600 | ∞ | 0 |
| 0.01 | 0.040 | 1.600 | 40 | 1 |
| 0.05 | 0.200 | 1.600 | 8 | 1 |
| 0.10 | 0.400 | 1.600 | 4 | 1 |
| 0.40 | 1.600 | 1.600 | 1 | 1 |

*(a/R = 0.2, Poiseuille, units of `v_mean/R`)*

The crossover is at `|e| = 2a`. **The scalar aggregate only becomes as sensitive
as the differential once the catheter has been displaced by twice its own radius
— by which point it is against the wall and the question is moot.**

### 3.2 Chain rule into the existing stack

The measurement Jacobian with respect to *configuration* factors cleanly:

```
H = ∂s/∂χ = M · P · J
    ( 4×7 )  (4×2)(2×3)(3×7)
```

where `J = ∂p/∂χ` is **the existing 3×7 beam-position Jacobian, unchanged**,
`P` projects tip position onto the local transverse plane of the centreline, and
`M` is the flow map above. Since `rank H ≤ min(rank M, rank J)`:

* `rank M` is a **hard ceiling** on the whole chain, and it is pure flow physics —
  no robot model needed to evaluate it.
* Therefore the cheap analysis (§6) is a genuine gate. If `M` is rank-deficient,
  nothing downstream can help, and no beam simulation need be run to find out.
* The existing Jacobian provider infrastructure is reused verbatim. The
  integration surface is one new 4×2 block.

This is why the work plan evaluates `M` alone first and only then composes with `J`.

---

## 4. Pulsatile flow

### 4.1 Womersley solution

For an oscillatory pressure gradient in a rigid tube, with `y = r/R`,
`β = α i^{3/2}` and

```
α = R √(ωρ/μ)
```

the axial velocity is

```
u(y,t) = Re{ (iP/(ρω)) [1 − J₀(βy)/J₀(β)] e^{iωt} }
```

The quantity that matters is the radial gradient at the sensor ring, normalised
by mean flow so it is dimensionless and comparable across vessels. Define the
**discrimination gain**

```
Γ(α, a/R) = R · |∂u/∂r|_{r=a} / |v_mean|
          = |β J₁(β a/R) / J₀(β)| / |1 − 2J₁(β)/(β J₀(β))|
```

**Poiseuille limit.** As `α → 0`, `Γ → 4a/R`, recovering §3.1 exactly. Verified
numerically to a relative error of 2.8 × 10⁻¹⁰ — this is the implementation
check the code must reproduce.

### 4.2 The gain collapses with α

`Γ` normalised by its Poiseuille value `4a/R` (1.00 = as good as steady flow):

| α | a/R=0.1 | a/R=0.2 | a/R=0.3 | a/R=0.5 |
|---|---|---|---|---|
| 1 | 0.999 | 0.999 | 0.999 | 0.999 |
| 2 | 0.986 | 0.986 | 0.987 | 0.989 |
| 3 | 0.935 | 0.935 | 0.937 | 0.947 |
| 4 | 0.824 | 0.824 | 0.828 | 0.857 |
| 5 | 0.664 | 0.665 | 0.672 | 0.729 |
| 7 | 0.348 | 0.351 | 0.365 | 0.472 |
| 10 | 0.096 | 0.100 | 0.115 | 0.223 |
| 15 | 0.008 | 0.009 | 0.014 | 0.056 |
| 20 | 0.000 | 0.001 | 0.002 | 0.013 |

**Mechanism.** As α rises the Stokes layer `δ ≈ √(2ν/ω)` thins and the core moves
as a plug. A plug has no radial gradient, so there is nothing for a differential
measurement to detect. The knee is at **α ≈ 4–5**; by α ≈ 10 the gain has fallen
by an order of magnitude, and by α ≈ 15 by two.

Note the weak dependence on `a/R` compared with the strong dependence on `α`.
Pushing the sensors outward helps only marginally, and only at high α where the
ring may reach into the Stokes layer. **The catheter cannot be redesigned out of
this problem.**

### 4.3 Real vessels (blood, ρ = 1060, μ = 3.5 mPa·s, 72 bpm, a = 1.15 mm)

| vessel | R (mm) | α | a/R | Γ/Γ_P | Γ |
|---|---|---|---|---|---|
| radial | 1.3 | 1.96 | 0.423 | 0.989 | 1.673 |
| coronary | 1.5 | 2.27 | 0.367 | 0.979 | 1.436 |
| carotid | 3.5 | 5.29 | 0.329 | 0.629 | 0.826 |
| femoral | 4.0 | 6.04 | 0.287 | 0.499 | 0.574 |
| iliac | 5.5 | 8.31 | 0.209 | 0.210 | 0.176 |
| descending aorta | 10.0 | 15.11 | 0.115 | 0.007 | 0.003 |
| ascending aorta | 14.0 | 21.16 | 0.082 | 0.000 | 0.000 |

### 4.4 Phase structure — the observability null is not the flow null

Both `Ĝ` (gradient) and `V̂` (mean flow) are complex, and **their arguments
differ**. The phase separation between the instant of zero gradient and the
instant of zero mean flow:

| α | 2 | 3 | 5 | 10 |
|---|---|---|---|---|
| null separation (deg) | 17.9 | 39.6 | 100.4 | 92.5 |

**Consequence for the estimator.** You cannot infer "the measurement is
uninformative right now" from a flowmeter reading — at α = 5 the gradient nulls
roughly a quarter-cycle away from the flow null. Observability gating must be
driven by the *phase-aware model*, not by measured flow magnitude. Getting this
backwards would gate the estimator at precisely the wrong moments.

### 4.5 Duty cycle — deliberately not claimed here

A pure single harmonic nulls twice per cycle regardless of α, so a single-harmonic
duty-cycle figure measures the sinusoid, not the physics. Physiological waveforms
carry a steady component plus several harmonics, and the honest duty cycle depends
on the pulsatility index and the harmonic content.

Preliminary single-harmonic sweeps suggest the worst case is **PI ≈ 1–1.5**, where
the oscillatory component nearly cancels the steady component at some phase without
being large enough to dominate elsewhere. This is flagged as a *hypothesis to test
with a real waveform*, not a result. The work plan (D3) computes it properly.

---

## 5. Bend-induced bias

### 5.1 The flow maximum is not the geometric centreline

In a curved tube, centrifugal effects drive the core outward and set up
counter-rotating Dean vortices. The velocity peak shifts **toward the outer
wall**. Governed by

```
De = Re √(R/R_c),     Re = 2 R v_mean ρ / μ
```

**This is the most dangerous failure mode in the project.** An estimator that
assumes an axisymmetric profile centred on the geometric centreline will read the
outward peak shift as a lateral offset and command the catheter *outward* — into
the wall it is meant to avoid, and precisely where contact modelling and the
existing contact-Jacobian mismatch work becomes active. It is a bias, not noise,
so it does not average out.

Dean numbers for representative geometries:

| vessel | Re | R_c=10mm | 25mm | 50mm | 100mm |
|---|---|---|---|---|---|
| coronary | 136 | 53 | 33 | 24 | 17 |
| carotid | 636 | 376 | 238 | 168 | 119 |
| femoral | 606 | 383 | 242 | 171 | 121 |
| iliac | 999 | 741 | 469 | 331 | 234 |

Rough bands: `De < 20` weak (shift ≲ 2% R); `20–100` developing; `>100` strong
twin vortices with shifts reported at 10–20% R. **These magnitudes are literature
correlations, not derived here** — see §8.

Only the coronary case sits reliably in the weak band. Everywhere else the bend
correction is mandatory, and the existing `LumenBend` angles feed it directly.

### 5.2 The quadrupole channel detects it — with one confound

Dean secondary flow is not axisymmetric, so it drives `s₀ − s₁ + s₂ − s₃` away
from zero. The array carries its own model-validity monitor at no hardware cost.

But the null is weaker than §2.2 suggests, and this correction matters. Expanding
to **second** order in the offset gives

```
Q = [ u″(a) − u′(a)/a ] (e_y² − e_z²) + O(e⁴)
```

The bracket measures departure from parabolic at the sensor ring, and **vanishes
identically for any `u = A + Br²`**. So:

* Under **Poiseuille**, `Q ≡ 0` exactly, at any offset — verified to 1e-14.
* Under **Womersley**, the profile is a Bessel function, the bracket is non-zero,
  and an off-centre catheter generates a quadrupole signal *even with no secondary
  flow at all*. Verified against the numerics to within 2%.

Magnitude, as a fraction of the differential signal at the same offset:

| α | \|e\|/R = 0.02 | 0.05 | 0.10 | 0.15 |
|---|---|---|---|---|
| 2 | 0.05% | 0.13% | 0.25% | 0.38% |
| 5 | 0.62% | 1.56% | 3.11% | 4.66% |
| 10 | 0.42% | 1.06% | 2.12% | 3.16% |

**Consequence.** The quadrupole is a clean Dean detector only near the centreline.
At `|e| ≤ 0.05R` the confound sits below ~1.5% of the differential signal and is
usually under the noise floor; by `|e| = 0.15R` at α = 5 it reaches ~5% and would
be misread as secondary flow. Correct use is therefore a **residual test**:
predict `Q` from the current offset estimate and the known profile, subtract, and
attribute only what remains to Dean flow. It is a diagnostic on the residual, not
a direct reading — which is also why it cannot be used to bootstrap an offset
estimate it depends on.

Note the peak at α ≈ 5: the confound is worst where the profile is furthest from
*both* parabolic and plug. That is squarely the carotid/femoral range.

---

## 6. Feasibility criterion

Differencing two sensors gives noise `√2 σ_s` against gain `2Γ v_mean/R`, so the
smallest resolvable offset is

```
δe/R = (1/√2) · (σ_s / v_mean) / Γ(α, a/R)
```

All the flow physics sits in `Γ`. Taking a tolerance of `δe/R ≤ 0.10` and a 7 Fr
catheter (`a = 1.15 mm`) at 72 bpm:

| R (mm) | α | Γ | blockage | δe/R @1% | δe/R @2% | verdict |
|---|---|---|---|---|---|---|
| 2.0 | 3.02 | 2.197 | 33.1% | 0.003 | 0.006 | OK |
| 3.0 | 4.53 | 1.165 | 14.7% | 0.006 | 0.012 | OK |
| 4.0 | 6.04 | 0.574 | 8.3% | 0.012 | 0.025 | OK |
| 5.0 | 7.56 | 0.264 | 5.3% | 0.027 | 0.054 | OK |
| 6.0 | 9.07 | 0.116 | 3.7% | 0.061 | 0.122 | MARGINAL |
| 8.0 | 12.09 | 0.020 | 2.1% | 0.349 | 0.698 | FAIL |
| 10.0 | 15.11 | 0.003 | 1.3% | 2.16 | 4.32 | FAIL |

The boundary sits near **R ≈ 5–6 mm** and moves slowly with sensor quality. Solving
`δe/R = 0.10` exactly for the boundary radius:

| catheter | σ_s/v_mean = 1% | 2% | 5% |
|---|---|---|---|
| 3 Fr (a = 0.55 mm) | 5.68 mm | 4.83 mm | 3.62 mm |
| 7 Fr (a = 1.15 mm) | 6.58 mm | 5.76 mm | 4.63 mm |

A **5× noise improvement buys about 2 mm** of vessel radius, and doubling the
catheter radius buys about **1 mm**. Both are weak levers against a quantity that
falls by two orders of magnitude between α = 5 and α = 15. **The limit is the flow
profile, not the instrument** — which also means effort spent on sensor quality
has a low ceiling, and is worth knowing before it is spent.

Read as anatomy: comfortable in coronary, radial and other peripheral vessels;
usable in carotid and femoral; marginal at the iliac; infeasible in the aorta.

---

## 7. Where the model stops being valid

The expansion in §2.2 samples an *undisturbed* profile. The catheter occupies
`(a/R)²` of the lumen area, and at the small end of the range that is not small:

| vessel | a/R | blockage |
|---|---|---|
| radial | 0.423 | 17.9% |
| coronary | 0.367 | 13.4% |
| carotid | 0.329 | 10.8% |
| femoral | 0.287 | 8.3% |
| iliac | 0.209 | 4.4% |

**The undisturbed-profile assumption is weakest exactly where the method works
best.** Above ~10% blockage the correct model is flow in an *eccentric annulus*,
where the fluid redistributes toward the wide gap.

That redistribution has the **same sign** as the effect being exploited — the
sensor on the side away from the wall still reads faster — so the qualitative
conclusion is safe and the `Γ` values above are probably **conservative**. But
the *gain* is wrong, and a gain error propagates directly into estimator bias.
Calibrating `Γ` against an eccentric-annulus solution (or CFD) at `a/R > 0.3` is a
required task, not an optional refinement.

---

## 8. Status of every claim

Stated explicitly so the experimental work knows what it is testing versus assuming.

**Exact, derived here, reproducible by the code:**

* channel decomposition (§2.2); the quadrupole null for parabolic flow and its
  second-order offset term under Womersley (§5.2)
* scalar rank-1 / differential rank-2 structure; `4|e|/R` vs `8a/R` (§3.1)
* `H = M·P·J` factorisation and the rank ceiling (§3.2)
* the Womersley `Γ(α, a/R)`, its Poiseuille limit to 2.8e-10, and the α-collapse (§4)
* phase separation of gradient and flow nulls (§4.4)
* the resolution formula and feasibility envelope (§6)

**Scaling arguments — direction trustworthy, magnitude not:**

* Dean peak-shift magnitudes (§5.1); bands are literature correlations
* the eccentric-annulus sign argument (§7)

**Requires calibration before any number is quoted:**

* `σ_s` and the sensor transfer function — everything in §6 scales with `σ_s/v_mean`
* whether the sensors read speed, a thermal proxy, or something with its own dynamics
* the physiological waveform (harmonic content and PI) driving §4.5
* `Γ` at `a/R > 0.3`, against an eccentric-annulus model

---

## 9. Connection to existing code

Deliberately minimal. The control ladder, planner and beam Jacobian are untouched.

| new | mirrors | note |
|---|---|---|
| `flow/flow_field.py` | — | Womersley + steady, parameterised by `(R, ω, ρ, μ, waveform)` |
| `flow/sensor_model.py` | — | `h(χ) → 4 channels`; ring geometry `(a, φ_k)` a parameter |
| `flow/flow_jacobian.py` | `beam_jacobian_providers.py` | provider pattern **with provenance**; `M` by central differences on the same convention as the existing 3×7 |
| `flow/pipelines.py` | `contact_and_contact_free()` | returns `scalar_aggregate` and `differential` as two providers over one model — the direct analogue, and the controlled comparison |
| `analysis/compare_sensor_pipelines.py` | `compare_providers()` | cheap, no closed loop — the D1 gate |
| `analysis/analyse_layer5_flow.py` | `analyse_layer4_mpc.py` | same report/metrics/PNG triple, PASS/WARN/FAIL/NOTE, `budget = error/tolerance`, `--self-test` |

Reused unchanged: `LumenConfig` / `LumenBend` (radius and bend angle now also
drive `α` and `De`), the L1–L3 centreline as the reference, the beam Jacobian
providers, and `_stack_common.py` helpers.

**The six-point geometry cross carries over intact.** Radius and bend angle
acquire second meanings, which is what makes the reuse worth something:

| axis | old meaning | new meaning |
|---|---|---|
| lumen radius | contact severity | `α` and `a/R` → **discrimination gain** |
| bend angle | scheduling severity `S` | `De` → **Dean bias** |

The cross still isolates them, and the predictions in §10 are stated per arm.

---

## 10. Experimental programme — cheap analysis only

Every item below is evaluated from the flow model and geometry. No beam
simulation, no QP solve, no closed loop. Each states its prediction first, so a
disagreement is informative rather than embarrassing.

**D0 — implementation gate.** `Γ(α→0) = 4a/R` to < 1e-8 relative; quadrupole
residual < 1e-12 for any axisymmetric profile; `M` by central differences agrees
with the analytic `8a/R` to < 1e-6. *Nothing downstream is meaningful until these
pass.*

**D1 — scalar vs differential conditioning.** *Prediction:* `σ_min` of the scalar
map falls linearly to zero as `|e| → 0` and its rank is 1; the differential map
holds `σ_min = 8a/R` with rank 2 at all offsets; they cross at `|e| = 2a`.
*This is the headline plot and the justification for the data-path change.*

**D2 — discrimination gain vs Womersley number.** Sweep α over the six-point
cross. *Prediction:* `Γ/Γ_P` follows the §4.2 table — flat below α ≈ 2, knee at
4–5, an order of magnitude down by α ≈ 10. *Prediction:* varies strongly along the
radius arm of the cross and is **flat along the bend-angle arm**, since α depends
on `R` alone. That flatness is the isolation check.

**D3 — duty cycle with a real waveform.** Replace the single harmonic with a
measured or literature flow waveform. *Prediction:* observability is retained for
most of the cycle where a steady component is present; worst case near PI ≈ 1–1.5.
*Prediction:* the low-gain window is offset in phase from the low-flow window by
the §4.4 separation. Report the window duration — it sets how long an estimator
must coast, which is the input to the estimator design that follows later.

**D4 — Dean bias vs bend angle.** *Prediction:* apparent lateral offset grows with
`De`, is directed toward the outer wall, is **flat along the radius arm** of the
cross and varies strongly along the bend-angle arm — the mirror image of D2, which
is what makes the cross worth running. Report bias in units of `R` against the
0.10 `R` tolerance.

**D5 — quadrupole residual as a Dean detector.** *Prediction:* residual is zero to
numerical precision under axisymmetric flow and grows with `De`. *Test:* is it
above `√2 σ_s` at realistic noise? If yes, the array self-diagnoses its own model
violation; if no, say so plainly.

**D6 — feasibility envelope.** Compose D2, D3 and D4 into a map over
`(R, bend angle, σ_s/v_mean)` graded PASS / MARGINAL / FAIL against `δe/R ≤ 0.10`,
with the Dean bias added to the noise-driven resolution. *Prediction:* boundary at
`R ≈ 5–6 mm` set by α, moving < 1 mm per 5× change in sensor noise. **This map is
the deliverable.**

**Then stop.** No estimator, no controller, no closed loop until this is reviewed.

---

## 11. What comes after, and why it is not specified yet

The estimator design follows from D3 and D4, and specifying it earlier would be
guessing:

* If the D3 low-gain window is short relative to the beam's transverse settling
  time, a filter that coasts through it suffices.
* If it is long, the estimator must be gated on predicted observability, and the
  controller must hold rather than act on a stale estimate — which becomes a
  constraint in the existing MPC, not a new controller.
* If D4 shows Dean bias above tolerance, the feedforward flow model must carry the
  bend correction, and that correction's *accuracy* becomes a rung in its own right
  — the direct analogue of the contact/no-contact Jacobian mismatch already studied.

One structural point worth recording now: for nonlinear systems the separation
principle does not hold, so estimator rung and controller rung will interact.
Excitation is the reason — the differential channels are linear in `e` with
`e`-independent gain, so unlike a single-sensor scheme this problem does **not**
require dither for observability. That is a significant simplification and it is
worth stating clearly: **the four-sensor ring buys the removal of the dual-control
problem entirely.** Whether the residual nonlinearity (through `R(z)`, the bend
correction, and the beam model) reintroduces it is an open question for later.

There is also a free by-product of the common-mode channel. Since `Q` is conserved
along an unbranched segment, `v_mean = Q/(πR²)` means the mean reading estimates
the **local vessel radius** — which, against a preoperative map, gives longitudinal
localisation without vision. Not part of the feasibility claim, but it costs
nothing and is worth instrumenting while the machinery is being built.
