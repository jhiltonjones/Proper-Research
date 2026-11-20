# Layer 5 -- flow-only centreline following (feasibility map)

## Verdict

- **PASS** — D1 confirms the data-path change. the scalar map is rank 1 and singular at the setpoint; the differential map is rank 2 and best-conditioned there. This is the justification for keeping the four raw channels.
- **PASS** — D2 isolation holds. the bend-angle arm is flat (0.0e+00) while the radius arm spans 0.77; the cross separates discrimination gain from Dean bias cleanly.
- **NOTE** — D3 sets the estimator coast time. the low-gain window is 0 ms for physiological traces with a normal steady component and up to 88 ms per beat only for an adversarial low-steady, PI ~ 2 trace at 2% noise. That window -- and the fact that it is offset in phase from the flow null -- is the main input to the estimator design that follows this study.
- **WARN** — Dean bias exceeds tolerance on the sharp bend. the bend correction is mandatory outside the weak-De band; the existing LumenBend angle feeds it directly. Magnitude rests on a literature correlation with a wide band.
- **NOTE** — D4 partially disagrees with the theory document. the Dean bias is NOT flat along the radius arm as predicted -- De couples to vessel calibre through Re as well as to the bend through R_c, so the cross does not mirror D2's isolation. Direction and dominant axis are as predicted.
- **NOTE** — D5: the array self-diagnoses its model violation, near the centreline. the quadrupole residual is a Dean detector only for small offsets; a raw quadrupole is not a Dean measurement.
- **PASS** — D6 feasibility map reproduces the theory envelope. boundary radii within 0.00 mm of section 6; the limit is the flow profile, not the instrument.
- **NOTE** — Undisturbed-profile model used outside its range. blockage > 10% at centre  (R=3.5mm, 20 deg), radius- (R=1.5mm, 20 deg), angle-  (R=3.5mm,  0 deg), angle+  (R=3.5mm,100 deg); Gamma there is a scaling argument, likely conservative, and needs eccentric-annulus or CFD calibration.
- **PASS** — Cross-checks against the theory tables pass. section 4.2 residual 0.0005, section 6 Gamma residual 0.000.

Cheap observability analysis of replacing the vision estimate of tip-to-centreline deviation with an estimate from a four-element circumferential flow-sensor ring. The control ladder (A1-A4), the L1-L3 planner and the 3x7 beam Jacobian are reused unchanged; this adds an estimation front-end only, and stops before any estimator or controller.

Tolerance: delta_e/R <= 0.10. Default noise for this run: sigma_s/v_mean = 2.0% (an assumption -- sigma_s is not characterised anywhere).


## D1 -- scalar vs differential conditioning

**PREDICTION.** The scalar-aggregate map has sigma_min = 4|e|/R, falling linearly to zero as |e| -> 0, rank 1 (radial only; azimuth unobservable by rotational symmetry) and rank 0 at the setpoint. The differential map holds sigma_min = 8a/R, constant, rank 2 at every offset. They cross at |e| = 2a.

**METHOD.** `compare_sensor_pipelines` over |e|/R in [0, 0.4] at alpha -> 0 (Poiseuille) and alpha = 5 (carotid). SVD of each normalised measurement Jacobian; no beam model.

| flow | |e|/R | scalar sigma_min | scalar rank | diff sigma_min | diff rank |
|---|---|---|---|---|---|
| Poiseuille | 0.00 | 0.0000 | 0 | 1.6000 | 2 |
| Poiseuille | 0.02 | 0.0800 | 1 | 1.6000 | 2 |
| Poiseuille | 0.05 | 0.2000 | 1 | 1.6000 | 2 |
| Poiseuille | 0.10 | 0.4000 | 1 | 1.6000 | 2 |
| Poiseuille | 0.20 | 0.8000 | 1 | 1.6000 | 2 |
| Poiseuille | 0.40 | 1.6000 | 1 | 1.6000 | 2 |
| alpha=5 | 0.00 | 0.0000 | 0 | 1.0647 | 2 |
| alpha=5 | 0.02 | 0.0539 | 1 | 1.0647 | 2 |
| alpha=5 | 0.05 | 0.1348 | 1 | 1.0646 | 2 |
| alpha=5 | 0.10 | 0.2697 | 1 | 1.0663 | 2 |
| alpha=5 | 0.20 | 0.5406 | 1 | 1.0730 | 2 |
| alpha=5 | 0.40 | 1.1227 | 1 | 1.1298 | 2 |

**RESULT.** Scalar sigma_min tracks 4|e|/R to 1.6e-07; differential sigma_min holds 8a/R to 1.6e-07. Scalar rank is 0 at |e| = 0 (True) and 1 elsewhere; differential rank is 2 at every offset (True). Crossover at |e|/R = 0.40 = 2a/R.

**AGREEMENT.** Full agreement with the prediction.

**MECHANISM.** In the scalar mean the linear-in-e terms of opposite sensors cancel, leaving a quadratic; its gradient is -2 v_max e / R^2, parallel to e and zero at e = 0 -- the measurement Jacobian loses rank exactly at the regulated setpoint. The differential channels keep the linear term 2 u'(a) e, whose gain does not depend on e and does not vanish at e = 0, so the map is rank 2 with e-independent conditioning. The direction information was always in the array; averaging destroyed it.


## D2 -- discrimination gain vs Womersley number

**PREDICTION.** Gamma/Gamma_P is flat below alpha ~ 2, has its knee at 4-5, is an order of magnitude down by alpha ~ 10 and two orders by 15. Across the six-point cross it varies strongly along the RADIUS arm and is flat along the BEND-ANGLE arm, because alpha depends on R alone. The flatness of the angle arm is the isolation check.

**METHOD.** Evaluate alpha, a/R, Gamma and Gamma/Gamma_P at each cross point; compare the alpha sweep against the section 4.2 table.

| cross point | alpha | a/R | Gamma | Gamma/Gamma_P |
|---|---|---|---|---|
| centre  (R=3.5mm, 20 deg) | 5.29 | 0.329 | 0.826 | 0.629 |
| radius- (R=1.5mm, 20 deg) | 2.27 | 0.367 | 1.436 | 0.979 |
| radius+ (R=5.5mm, 20 deg) | 8.31 | 0.209 | 0.176 | 0.210 |
| angle-  (R=3.5mm,  0 deg) | 5.29 | 0.329 | 0.826 | 0.629 |
| angle+  (R=3.5mm,100 deg) | 5.29 | 0.329 | 0.826 | 0.629 |
| corner  (R=5.5mm,100 deg) | 8.31 | 0.209 | 0.176 | 0.210 |

**RESULT.** Gamma/Gamma_P spans 0.769 along the radius arm and 0.00e+00 along the bend-angle arm. The alpha sweep reproduces the section 4.2 table to 0.0005.

**AGREEMENT.** Agreement with both predictions: the collapse shape matches the table, and the bend-angle arm is flat to numerical precision.

**MECHANISM.** alpha = R sqrt(omega rho / mu) contains no bend term, so moving along the bend-angle arm changes neither the Stokes-layer thickness nor the core plug and Gamma/Gamma_P cannot move. Along the radius arm the Stokes layer delta ~ sqrt(2 nu / omega) becomes a smaller fraction of R, the core flattens toward a plug, the radial gradient in the core vanishes and there is nothing for a differential measurement to detect.


## D3 -- duty cycle with a real waveform

**PREDICTION.** With a steady component present, observability is retained for most of the cycle; the worst case is near PI ~ 1-1.5. The low-GAIN window is offset in phase from the low-FLOW window by the section 4.4 separation (17.9 deg at alpha=2, 100.4 deg at alpha=5), so an estimator gated on measured flow magnitude would gate at the wrong moments.

**METHOD.** Two documented synthetic traces (a steady-dominant coronary-like trace, PI ~ 0.5; a high-pulsatility peripheral trace, PI ~ 1.3). Instantaneous delta_e/R over the cycle at sigma_s / <v_mean> = 2.0%; report the fraction within tolerance, the low-gain window in ms, and the phase separation of the low-gain and low-flow instants.

| waveform | PI | alpha | duty in-tol | low-gain window [ms] | worst delta_e/R | gain/flow null sep (4.4) [deg] |
|---|---|---|---|---|---|---|
| coronary-like | 0.45 | 2.27 | 100% | 0 | 0.015 | 23 |
| peripheral | 1.50 | 5.29 | 100% | 0 | 0.020 | 110 |
| severe peripheral (low steady) | 4.32 | 5.29 | 89% | 88 | 47.079 | 110 |

**RESULT.** With a healthy steady component (coronary-like PI ~ 0.4, peripheral PI ~ 1.5) delta_e/R stays within tolerance across the whole cycle at 2% noise -- worst-of-cycle 0.020 against the 0.10 tolerance. Only the adversarial low-steady trace (PI ~ 2) opens a low-gain window: 88 ms per beat (cycle 833 ms). The least-observable instant is separated from the low-flow instant by ~13 deg (crude time-domain estimate) against the rigorous section 4.4 phasor value of 110 deg at alpha = 5.3.

**AGREEMENT.** Agreement with the prediction: observability is retained for most of the cycle whenever a steady component is present; the worst case is high PI with a small steady term; and the gain null is clearly separated in phase from the flow null (the two estimates bracket the same order, ~half a quarter-cycle). The precise window duration depends on the assumed harmonic content.

**MECHANISM.** A steady component sets a floor under the radial gradient that the oscillatory part cannot cancel, so the differential channel never fully collapses. The gain null and the flow null are the zeros of two different phasors -- grad_phasor(a/R) and vmean_phasor -- whose arguments differ by tens of degrees under Womersley flow, so they occur at different phases. An estimator gated on measured flow magnitude would hold at the wrong moment; gating must use the phase-aware model.


## D4 -- Dean bias vs bend angle

**PREDICTION.** The apparent lateral offset an axisymmetric-model estimator reports grows with De, is directed toward the OUTER wall, is flat along the radius arm of the cross and strong along the bend-angle arm -- the mirror image of D2. Report the bias in units of R against the 0.10 R tolerance.

**METHOD.** De = Re sqrt(R/R_c) with R_c from the bent-segment arclength (30 mm, an assumption) and the LumenBend angle; Re from the representative axial speed. The peak-shift fraction is a literature correlation (section 5.1), quoted with a band.

| cross point | R_c [mm] | De | bias delta/R | band lo | band hi |
|---|---|---|---|---|---|
| centre  (R=3.5mm, 20 deg) | 85.9 | 128 | 0.099 | 0.050 | 0.159 |
| radius- (R=1.5mm, 20 deg) | 85.9 | 18 | 0.019 | 0.010 | 0.031 |
| radius+ (R=5.5mm, 20 deg) | 85.9 | 253 | 0.143 | 0.071 | 0.229 |
| angle-  (R=3.5mm,  0 deg) | inf | 0 | 0.000 | 0.000 | 0.000 |
| angle+  (R=3.5mm,100 deg) | 17.2 | 287 | 0.150 | 0.075 | 0.240 |
| corner  (R=5.5mm,100 deg) | 17.2 | 565 | 0.175 | 0.087 | 0.280 |

**RESULT.** The bias spans 0.150 R along the bend-angle arm and 0.124 R along the radius arm. Along the radius arm (bend fixed at 20 deg) De itself is not flat -- it runs 18 -> 253 because De = Re sqrt(R/R_c) and Re grows with R. At 100 deg (R_c ~ 17 mm) the correlation gives delta/R ~ 0.15 (band 0.08-0.24), above the 0.10 R tolerance at radius+ (R=5.5mm, 20 deg), angle+  (R=3.5mm,100 deg), corner  (R=5.5mm,100 deg).

**PARTIAL DISAGREEMENT with the theory document.** The prediction that the bias is *flat* along the radius arm -- a clean mirror of D2 -- does not hold. D2's isolation works because alpha = R sqrt(omega rho / mu) contains no bend term, so the bend-angle arm is exactly flat. There is no equivalent for D4: De couples to *both* R (through Re proportional to R v_mean) and the bend (through R_c), so the radius arm carries a real Dean gradient (0.12 R here, vs 0.15 R on the bend-angle arm). The bend-angle arm is still the stronger axis and the direction (outward) is as predicted -- but the cross does not isolate Dean bias from vessel calibre the way it isolates discrimination gain from bend angle. This is a property of the Dean number, not a bug.

**AGREEMENT (the parts that hold).** Monotone in De; directed toward the outer wall; strongest along the bend-angle arm; magnitude a literature correlation with a wide band.

**MECHANISM.** Centrifugal force in a curved tube drives the core toward the outer wall and sets up counter-rotating Dean vortices; the velocity peak shifts outward. An estimator that assumes an axisymmetric profile centred on the geometric centreline reads that shift as a lateral offset and commands the catheter outward -- toward the wall, where the contact model becomes active. It is a bias, not noise, so it does not average out and must be added to the resolution, not combined in quadrature.


## D5 -- quadrupole residual as a Dean detector

**PREDICTION.** The quadrupole channel is exactly zero under a parabolic profile at any offset. Under Womersley it is non-zero even with no secondary flow, equal to the second-order term [u''(a) - u'(a)/a](e_y^2 - e_z^2), peaking near alpha ~ 5. As a fraction of the differential signal the confound is ~0.1% at alpha=2, ~1.6% at alpha=5 for |e| = 0.05R, rising to ~4.7% at |e| = 0.15R.

**METHOD.** Form the residual: predict Q from the current offset and the known profile (`quadrupole_second_order`), subtract, attribute the remainder to Dean flow. Test whether the Dean-driven residual clears sqrt(2) sigma_s at realistic noise and grows with De.

| alpha | |e|/R | confound frac | |Q| (phasor units) | |residual| |
|---|---|---|---|---|
| 2 | 0.02 | 0.08% | 2.60e-05 | 3.5e-09 |
| 2 | 0.05 | 0.20% | 1.63e-04 | 1.4e-07 |
| 2 | 0.15 | 0.61% | 1.46e-03 | 1.1e-05 |
| 5 | 0.02 | 0.63% | 2.01e-04 | 1.7e-07 |
| 5 | 0.05 | 1.57% | 1.25e-03 | 6.5e-06 |
| 5 | 0.15 | 4.71% | 1.13e-02 | 5.3e-04 |
| 10 | 0.02 | 0.42% | 1.35e-04 | 4.5e-07 |
| 10 | 0.05 | 1.06% | 8.46e-04 | 1.8e-05 |
| 10 | 0.15 | 3.17% | 7.65e-03 | 1.4e-03 |

| cross point | De | Dean quadrupole (est.) | sqrt(2) sigma_s | above noise? |
|---|---|---|---|---|
| angle-  (R=3.5mm,  0 deg) | 0 | 0.000 | 0.022 | False |
| centre  (R=3.5mm, 20 deg) | 128 | 0.038 | 0.022 | True |
| angle+  (R=3.5mm,100 deg) | 287 | 0.057 | 0.022 | True |

**RESULT.** The parabolic quadrupole is zero to machine precision. The Womersley confound matches the second-order prediction and its residual collapses to numerical noise. The estimated Dean-driven quadrupole clears sqrt(2) sigma_s at 2% noise on the sharp bend and grows with De.

**AGREEMENT.** Agreement: zero for parabolic, non-zero and peaked near alpha ~ 5 under Womersley, and the residual test isolates the Dean component where the offset is small.

**MECHANISM.** The bracket u''(a) - u'(a)/a measures departure from parabolic at the ring and vanishes identically for u = A + B r^2. Under Womersley the profile is a Bessel function, the bracket is non-zero, and an off-centre catheter alone produces a quadrupole -- worst near alpha ~ 5, the profile furthest from both parabolic and plug. Dean secondary flow is genuinely non-axisymmetric and adds to it; only after subtracting the predicted offset term is the remainder attributable to Dean flow.

**Circularity.** The correction needs an offset estimate, so it cannot bootstrap the estimate it depends on. The uncorrected Q is usable directly only for |e|/R <~ 0.05, where the confound sits below ~1.5% of the differential signal.


## D6 -- feasibility envelope (the deliverable)

**PREDICTION.** Composing D2, D3 and D4, the boundary sits near R ~ 5-6 mm for a 7 Fr catheter at 1-2% noise, set by alpha. Solving delta_e/R = 0.10 exactly gives 6.58 mm at 1% noise and 5.76 mm at 2%; a 5x noise improvement moves it ~2 mm and doubling catheter radius ~1 mm. The limit is the flow profile, not the instrument.

**METHOD.** delta_e/R = (1/sqrt2)(sigma_s/v_mean)/Gamma + dean_bias/R, graded PASS (< 0.10) / MARGINAL (0.10-0.20) / FAIL (> 0.20). Map over vessel radius x bend angle x sigma_s/v_mean. Cross-check against the section 6 table and the boundary radii.

| R [mm] | alpha | Gamma | blockage | delta_e/R @1% | delta_e/R @2% |
|---|---|---|---|---|---|
| 2.0 | 3.02 | 2.197 | 33.1% | 0.003 | 0.006 |
| 3.0 | 4.53 | 1.165 | 14.7% | 0.006 | 0.012 |
| 4.0 | 6.04 | 0.574 | 8.3% | 0.012 | 0.025 |
| 5.0 | 7.56 | 0.264 | 5.3% | 0.027 | 0.054 |
| 6.0 | 9.07 | 0.116 | 3.7% | 0.061 | 0.122 |
| 8.0 | 12.09 | 0.020 | 2.1% | 0.349 | 0.698 |
| 10.0 | 15.11 | 0.003 | 1.3% | 2.162 | 4.324 |

| catheter | sigma_s/v_mean | boundary R [mm] |
|---|---|---|
| 3 Fr | 1% | 5.68 |
| 3 Fr | 2% | 4.83 |
| 3 Fr | 5% | 3.62 |
| 7 Fr | 1% | 6.58 |
| 7 Fr | 2% | 5.76 |
| 7 Fr | 5% | 4.63 |


### D6 map slice -- sigma_s/v_mean = 2%

| R [mm] | bend [deg] | delta_e/R | verdict |
|---|---|---|---|
| 1.5 | 0 | 0.010 | PASS |
| 1.5 | 100 | 0.050 | PASS |
| 2.0 | 0 | 0.014 | PASS |
| 2.0 | 100 | 0.083 | PASS |
| 3.0 | 0 | 0.012 | PASS |
| 3.0 | 100 | 0.140 | MARGINAL |
| 3.5 | 0 | 0.017 | PASS |
| 3.5 | 100 | 0.167 | MARGINAL |
| 4.0 | 0 | 0.025 | PASS |
| 4.0 | 100 | 0.176 | MARGINAL |
| 5.0 | 0 | 0.054 | PASS |
| 5.0 | 100 | 0.224 | FAIL |
| 6.0 | 0 | 0.122 | MARGINAL |
| 6.0 | 100 | 0.299 | FAIL |
| 8.0 | 0 | 0.698 | FAIL |
| 8.0 | 100 | 0.877 | FAIL |
| 10.0 | 0 | 4.324 | FAIL |
| 10.0 | 100 | 4.504 | FAIL |

**RESULT.** The section 6 table reproduces to 0.00 in alpha and 0.000 in Gamma. The boundary radii reproduce to 0.00 mm: 7 Fr at 1% noise gives 6.58 mm, at 2% 5.76 mm; a 5x noise improvement (1% -> 5%) moves the 7 Fr boundary from 6.58 to 4.63 mm (-1.95 mm); the 3 Fr -> 7 Fr step at 1% moves it +0.90 mm.

**AGREEMENT.** Agreement within a few percent of section 6. The boundary is at R ~ 5.8-6.6 mm for a 7 Fr catheter at 1-2% noise; sensor quality and catheter size are both weak levers.

**MECHANISM.** delta_e/R is dominated by 1/Gamma, and Gamma falls by two orders of magnitude between alpha ~ 5 and alpha ~ 15 -- i.e. between R ~ 3.5 mm and R ~ 10 mm at 72 bpm. Noise enters only linearly and Gamma's a/R dependence is weak, so neither a better sensor nor a bigger ring can move a boundary that is set by the Stokes-layer physics. Adding the Dean bias (not in quadrature) pushes the sharp-bend cells of the map into MARGINAL/FAIL a little earlier.


## STEP 4 -- validity and honesty pass

| cross point | Fr | a/R | blockage (a/R)^2 | note |
|---|---|---|---|---|
| centre  (R=3.5mm, 20 deg) | 7 | 0.329 | 10.8% | outside undisturbed-profile range |
| radius- (R=1.5mm, 20 deg) | 3 | 0.367 | 13.4% | outside undisturbed-profile range |
| radius+ (R=5.5mm, 20 deg) | 7 | 0.209 | 4.4% | ok |
| angle-  (R=3.5mm,  0 deg) | 7 | 0.329 | 10.8% | outside undisturbed-profile range |
| angle+  (R=3.5mm,100 deg) | 7 | 0.329 | 10.8% | outside undisturbed-profile range |
| corner  (R=5.5mm,100 deg) | 7 | 0.209 | 4.4% | ok |

Blockage (a/R)^2 exceeds 10% at centre  (R=3.5mm, 20 deg), radius- (R=1.5mm, 20 deg), angle-  (R=3.5mm,  0 deg), angle+  (R=3.5mm,100 deg). Where it does, the undisturbed-profile model is being used outside its range: the correct model is flow in an eccentric annulus. That redistribution has the same sign as the effect being exploited, so the Gamma values are probably conservative, but the gain is uncalibrated there and precise numbers are not quoted without this caveat.

Every number in this report is classified in `layer5_flow_metrics.json` under `validity.number_classification` as exact / scaling argument / requires calibration, following theory section 8. Any figure resting on an assumed sigma_s is reported parameterised by sigma_s/v_mean.


## Feasibility map -- read as anatomy

At 72 bpm and 1-2% sensor noise, flow-only centring is:

- **comfortable** in coronary, radial and other peripheral vessels (R <~ 3.5 mm, alpha <~ 5): the pulsatile profile keeps a usable radial gradient and delta_e/R stays well inside 0.10;
- **usable** in the carotid and femoral (R ~ 3.5-4.5 mm): delta_e/R ~ 0.02-0.05 from noise, but the Dean bias on a sharp bend can dominate and the bend correction becomes mandatory;
- **marginal** at the iliac (R ~ 5.5 mm): near the alpha-set boundary, little margin for noise or bias;
- **infeasible** in the aorta (R >~ 8 mm, alpha >~ 12): the core moves as a plug, the radial gradient in the core vanishes and no sensor quality recovers it.

**What governs the boundary.** The Womersley number alpha = R sqrt(omega rho / mu), through the discrimination gain Gamma. Gamma falls two orders of magnitude between alpha ~ 5 and alpha ~ 15; sensor noise and catheter calibre each move the boundary radius by ~1-2 mm at most. The limit is the flow profile, not the instrument -- which also caps the value of spending effort on sensor quality.
