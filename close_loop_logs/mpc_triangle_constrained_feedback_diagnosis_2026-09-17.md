# Triangle constrained-feedback diagnosis — direct-apply LTV-MPC, 2026-09-17

## 1. Motivating question

Earlier work (`mpc_vs_inv_jacobian_2026-09-15.md`, `nullspace_motion_investigation_2026-09-16.md`)
established that MPC underperforms inverse-Jacobian control specifically on triangle, and that a
large, persistent, genuinely nonlinear model–plant residual (`d_phys`, ~5.6mm mean, `C_d(15)≈0.90`
persistence) correlates strongly with configuration departure `‖ξ‖` (partial corr 0.865 controlling
for path progress). What remained unresolved was *why the MPC's own feedback law fails to react
adequately to that error* — a question about the controller's cost/constraint structure, not the
disturbance model. This document is a from-scratch mechanistic audit of that question, run entirely
offline against a single validated live triangle run, using a numerically verified reconstruction of
the exact QP the controller solves.

Read together with `mpc_vs_inv_jacobian_2026-09-15.md` (the original comparison) and
`nullspace_motion_investigation_2026-09-16.md` (the null-space-content investigation that preceded
this one). This document supersedes both on the specific question of *feedback strength*.

## 2. A methodology correction discovered en route

Before any of the results below could be trusted, two bugs in the existing offline-replay machinery
had to be found and fixed:

- **`state_tracking_weight` was hardcoded to `0.001` in every prior offline replay script**
  (`exact_replay.py`, `dhat_decomposition.py`, `ablation_and_persistence.py`), sourced from a stale
  comment claiming it was "verified" against a run's `config.json`. It is not — `config.json`'s
  `mpc_state_tracking_weight: null` means *no override*, so the true live value is
  `ConfigurationMPCConfig`'s dataclass default, **`1.0`**, a 1000× difference. Fixed in all scripts
  used from this point forward.
- **The 2026-09-16 triangle log is not replayable for exact-QP analysis.** Re-running the
  exact-replay check (even after the weight fix) gave 30–70% relative error in the reconstructed
  `u0` and its task-space effect, far outside floating-point tolerance. Root cause: the live-logged
  `desired_mm` at a given `ref_index` does not correspond to *any* row of the plan file currently on
  disk (nearest match 1.9mm away). The plan file's mtime predates that run and hasn't changed since,
  so this isn't a case of the file being overwritten after the fact — the discrepancy's origin
  wasn't fully traced, and the run is being treated as **non-replayable for absolute QP analysis**.
  Its *geometry-only* findings (`‖d̂‖`, `d_phys`/`d_lin` decomposition, `C_d(j)`, the `‖ξ‖`↔`‖d̂‖`
  partial correlation) are unaffected, since those never touch the QP's `u0` at all.

Because of the second point, every result below comes from **fresh 2026-09-17 instrumented runs**,
not the 2026-09-16 log.

## 3. QP identity audit (the trust gate for everything that follows)

A `controller.solve` monkeypatch was used to capture the *exact* live-assembled `H, f, A, l, u` per
tick (in-RAM only, flushed to disk once after each run — verified not to perturb the hot loop's
wall-clock timing). Two short (40-tick), then one full-path (244-tick) triangle runs were audited.

- **Self-consistency**: audit-captured `u0` vs the independently-logged `u0` agree to
  `1.05e-05` mean, `5.47e-05` max relative error.
- **Level 4 — cold-started OSQP resolve of the dumped `H,f,A,l,u`, compared against the live `u0`**:
  `median ε_{u0} = 2.19e-14`, `max = 4.7e-5` — floating-point identity. The offline QP reconstruction
  is trustworthy for this session's runs.
- Solver health throughout: `status="solved"` on every audited tick, `primal_residual` typically
  `<1e-9`, `dual_residual` typically `<1e-2` — no convergence/iteration-limit contamination.

## 4. The apply-seam defect, and why the architecture is "direct-apply" not "no feedforward"

`feedforward_joint_trajectory` auto-disables for MPC kinds unless `force_mpc_feedforward=True` is
explicitly set (`close_loop_path_follow.py:675-687`, with an existing 2026-09-10 comment already
documenting "mpc_*+FF terminal error ~1.6mm" as the reason). Auditing *why*:

In FF mode, the applied joint target is `q_target = q_ref[k+1] + Δt·(u0*-u_ref[k+1])` — anchored to
the reference, not to the measured state — while the MPC's own dynamics model predicts
`z_{1|k} = z_meas + Δt·u0*`. Decomposing the seam residual `r_seam = r_anchor + r_clip` on a live
40-tick FF run: `‖r_anchor‖` (median-normalized ≈1.03) dominates `‖r_clip‖` (≈0.11, mostly exactly
zero) — the mismatch is structural reference-anchoring, not actuator clipping. This directly explains
the FF-mode terminal-error degradation and is now excluded from all further work.

The controller kind actually used for every result below is `mpc_ltv_offline` with
`feedforward_joint_trajectory=False`, i.e. `q_target = q_meas + Δt·u0*` (subject to the real 6mrad
step clip). This still contains the planner's feedforward motion **inside the optimization** via
`‖u-u_ref‖_R²` — it is not "MPC without feedforward," it is **direct-apply LTV-MPC**, the
terminology used from here on.

One further, separate finding from the same audit: **insertion has no hardware feedback at all.**
`insertion_length_getter=lambda: insertion_m` is a pure software-maintained variable
(dead-reckoned from `cfg.initial_insertion_m`), decoupled from whatever the real linear advancer
physically does. Flagged for future work, not addressed here.

## 5. The `F_k` / `κ` / `γ^F` framework

For the validated QP, `d̂` enters only the linear cost term (`H` is `d̂`-independent), which makes
both finite-difference and closed-form KKT sensitivity cheap and exact:

```
K_{e,k} = ∂u0*/∂e = -∂u0*/∂d̂          (since ∂d̂/∂e = -I)
F_k = Δt · J_{k+1} · K_{e,k}   ∈ R^{3x3}
κ_k = e_k^T F_k e_k / (‖e_k‖² + ε)
γ_k^F = ‖(I-F_k)e_k‖ / (‖e_k‖ + ε)
```

`κ≈1` and `γ^F→0` describe a controller that removes essentially all of the current error in one
step (the empirically successful INV-LTV-1.0 baseline's target regime). FD (perturbation
`δp∈{5,10,20}μm`) and the closed-form fixed-active-set KKT derivative agree to machine precision on
well-behaved ticks (median disagreement `~1e-12` to `9e-13`), with isolated disagreement only at
near-singular active-set transitions (flagged, not averaged in).

## 6. Constrained-regime results (full triangle path, real 0.06 rad/s + Δu + state bounds)

- **Every one of 222 usable ticks has at least one active joint-velocity-bound row** across the
  15-step horizon (`n_input_active` from 2 to 86). There is no free-tick subset in this run.
- Median `κ = 0.024`, median `γ^F = 0.982` — essentially no one-step contraction.
- `σ3(F) = 0.000` (median *and* p90) at every tick — a task-space direction has zero local
  first-command sensitivity under the real constraints.
- `σ1(F)` collapses to ~0.001–0.014 in the worst-error stretches (idx 121–141, 181–221, error
  10–12mm), vs ~0.35–0.87 in the benign early stretch — `κ` correlates `-0.634` with path progress.
- `ρ_authority` (best-possible geometric one-step correction fraction within the real 0.06 rad/s box,
  ignoring cost trade-offs): median `0.116`, falling to `0.022–0.086` exactly in the worst-error
  region, `corr(ρ_authority, error) = -0.732`. This is a genuine physical-authority scarcity,
  independent of any weight choice — treat "many ticks to recover" as intuition, not a strict bound,
  since `J`, the error direction and the reference all change tick to tick.
- Live rms on this run: **7.34mm** (transit), max 11.86mm — much worse than the looser-bound
  non-FF/direct-apply runs earlier this session (rms 3.9–4.1mm at 0.10 rad/s) or the FF runs
  (rms 1.60mm) or INV-LTV-1.0 (rms 0.93–1.10mm), consistent with the mechanism found here.

## 7. Isolating cost vs. constraint: bound-relaxation sweep

Re-solving the *same logged states* under `|u_q|≤{0.06, 0.10, 0.20, ∞}` (only the input box relaxed;
state/Δu bounds unchanged):

| bound | median κ | median γ^F | median σ3 | median n_input_active |
|---|---|---|---|---|
| 0.06 | 0.024 | 0.978 | 0.0000 | 21.0 |
| 0.10 | 0.137 | 0.933 | 0.0000 | 12.0 (n=35/55 — some KKT solves failed near this bound, caveat) |
| 0.20 | 0.049 | 0.964 | 0.0000 | 13.0 |
| ∞ (velocity box only) | 0.049 | 0.964 | 0.0000 | 13.0 |

Relaxing only the velocity box barely moves `γ^F`. This *initially* read as "the cost itself is
weak even unconstrained" — **but that conclusion was contaminated**: this sweep still enforced the
real `Δu` (rate/acceleration) and state-box constraints, which were independently found to be
materially binding. Re-run **truly unconstrained** (`H·v* = -f` directly, no bounds of any kind):
`κ ≈ 0.38–0.39`, `γ^F ≈ 0.64–0.67` at both a healthy and a bad tick — a substantially healthier
number. **The rate constraint, not the cost function, was responsible for most of the earlier
"weak-even-unconstrained" signature.** This correction is preserved here explicitly so it isn't
silently re-used.

## 8. KKT gradient-cancellation decomposition (truly unconstrained)

Writing `H = H_Q+H_R+H_{Rd}+H_{Qp}+H_f` (terminal), `f` likewise, and `g_i = H_i·v* + f_i`, the
unconstrained stationarity condition gives `Σg_i ≈ 0` exactly (`‖Σg_i‖ ~1e-10`). At all five audited
ticks (k=50 healthy; 125, 135, 190, 210 bad):

| term | `c_i = -g_i·g_Qp/(‖g_i‖‖g_Qp‖)` | `m_i = ‖g_i‖/‖g_Qp‖` |
|---|---|---|
| Q | +1.000 | 0.999 |
| R | +0.38–0.41 | 0.0002 |
| Rd | +0.21 | 0.0002 |
| terminal | +0.85–0.87 | 0.001 |

`Q` opposes `Qp` almost exactly, in both magnitude and direction; `R`, `Rd`, and the terminal term
are three to four orders of magnitude smaller. **Caveat, stated explicitly**: at a genuinely
unconstrained optimum with `R,Rd,V_f` negligible, `g_Q+g_{Qp}≈0` is close to a tautology of
first-order optimality, not on its own proof of a pathological fight — this is why the *causal*
tests in §9–§10 are the load-bearing evidence, not this decomposition alone.

**Structural verification, not just small-gradient inference**: removing the terminal cost's
`H,f` contribution and re-solving changes the unconstrained solution *only* in horizon steps 11–14;
steps 0–10, including `u0` itself, are unaffected to `1.4e-15` (machine precision), confirmed at two
independent ticks. `R→0.1R` and `Rd=0` produce task-space `u0` shifts of only 0.005–0.02mm.
**The terminal-cost/offset-equilibrium redesign considered earlier would not fix the weak-`u0`
problem — it structurally cannot reach `u0`.**

## 9. `σ3(F)` structural check

At `k=50,190`: `rank(F_e)=rank(K_e)=rank(F)=3` (full rank, no algebraic degeneracy in the
unconstrained sensitivity chain). The dead direction `r3` (right singular vector at `σ3`) aligns
with `J_{k+1}`'s **strongest**, best-conditioned left singular direction (`|r3·U_J[:,0]|=0.997` at
tick 50, `0.95` at tick 190), not its near-null geometric direction (`σ≈0.03–0.08`,
cosine `0.05–0.30`). **The controller's weak direction is not inherited from beam-Jacobian
ill-conditioning** — it is produced by the cost/constraint structure in a direction the kinematics
actually serve well.

## 10. `Q` and `Q_p` scalar sweeps — both falsified as simple fixes

**`Q` ablation** (`w_Q∈{1,0.1,0.01,0.001}`, bound fixed at 0.06): median `γ^F` moves `0.978→0.994`
(worse), `κ` `0.024→~0.010` (worse), active-input count `21→46` (worse). Reducing `Q` does not free
up authority for `Q_p` — if anything it removes a stabilizing effect and lets the optimizer chase
more constraints into saturation.

**`Q_p` sweep** (`w_{Qp}∈{0.25,0.5,1,2,4,10,20}`, full controller + terminal Riccati schedule
rebuilt each time, evaluated both truly-unconstrained and under the real 0.06/Δu/state constraints):

| `w_Qp` | σ3 (free) | σ3 (real) | κ (real), bad ticks 125/135/190/210 | κ (real), healthy tick 50 |
|---|---|---|---|---|
| 0.25 | 0.058 | ~0.000 | 0.013 / 0.019 / 0.011 / 0.005 | 0.143 |
| 1 | 0.199 | ~0.000 | 0.005 / 0.005 / 0.011 / 0.010 | 0.248 |
| 4 | 0.497 | ~0.000 | 0.000 / -0.000 / 0.001 / 0.000 | 0.308 |
| 20 | 0.832 | ~0.0001 | 0.000 / -0.000 / -0.000 / 0.000 | 0.329 |

`σ3` in the **unconstrained** problem grows strongly and monotonically with `w_Qp` — in isolation
this would say "just raise `Q_p`." But `σ3` under the **real** constraints stays at essentially
`0` across the entire 80× range, and `κ_real` at the bad ticks *degrades toward or below zero* as
`w_Qp` increases past ~2 (only the already-healthy tick 50 improves). A heavier `Q_p` makes the
unconstrained solution want a more extreme correction (`γ^F_free` exceeds 1.9 at `w_Qp=20`), which
drives *more* horizon-input rows into saturation (e.g. tick 210: `n_active` 27→89 across the sweep) —
exactly where the real feedback authority collapses.

`χ = ‖g_Q+g_{Qp}‖/‖g_{Qp}‖` stays at `0.0007–0.0014` across the entire 80× `w_Qp` range — per the
caveat in §8, this is expected near any unconstrained optimum and is not independent evidence by
itself, but combined with the causal sweep result (raising `Q_p` fails under real constraints, and
fails specifically where it's needed) it is consistent with `Q` and `Q_p` competing for the same
configuration authority at every scale tested, not merely being mis-scaled relative to each other.

## 11. Frozen conclusion

> The triangle failure is not attributable to Jacobian singularity, the DARE terminal term, `R`,
> `Rd`, a simple `Q`-weight error, or insufficient `Qp` magnitude alone. The unconstrained MPC has
> full-rank task sensitivity and substantially stronger feedback (`κ≈0.38`, `γ^F≈0.65`) than the
> constrained implementation, while the physically constrained controller loses one task-space
> feedback direction (`σ3→0`) and exhibits weak contraction (`γ^F≈0.98`) in the high-error portions
> of the trajectory. Scalar `Qp` increases strengthen the unconstrained response but increase
> horizon saturation and do not recover the constrained weak direction — in the bad-tick region they
> make it worse. Scalar reductions of `Q` likewise do not recover it. The evidence motivates
> restructuring the state cost so that task-useful configuration deviation and redundant/posture
> deviation are not penalized identically, rather than any further scalar weight search.

### Findings table

| Finding | Consequence |
|---|---|
| Exact replay: `ε_{u0}~2e-14` | Offline QP analysis is trustworthy |
| `rank(F_e)=rank(K_e)=rank(F)=3` | No intrinsic algebraic rank defect |
| True free MPC: `κ≈0.38-0.39`, `γ^F≈0.64-0.67` | Cost has meaningful feedback when genuinely unconstrained |
| Real 0.06 controller constrained on 222/222 triangle ticks | Constraints materially reshape the feedback law |
| Constrained `σ3(F)≈0` | A task-space direction becomes effectively unavailable under the active-set pattern |
| Dead direction aligns with `J`'s *strong* singular direction | Not caused by poor Jacobian conditioning |
| `ρ_authority` median 0.116, falling to 0.022-0.086 in the bad region | Genuine physical feedback scarcity also exists |
| `Vf` removal changes only horizon steps 11-14, not `u0` (to 1e-15) | Terminal redesign is not the immediate fix |
| `R,Rd` counterfactuals: 0.005-0.02mm effect on `u0` | Not primary levers |
| `Q↓1000×`: `γ^F` 0.978→0.994 (worse) | Simple state-weight reduction fails |
| `Qp↑80×`: free `σ3` recovers, real `σ3` stays ~0, bad-tick `κ` worsens | Simple task-weight increase fails under saturation |

## 12. Next: Stage-A Candidate 1 — task/posture-decomposed state cost

Not yet implemented. Design, to build next:

In dimensionless coordinates `δz̃_j = S_z⁻¹(z_{j|k}-z_{ref,k+j})`, `J̃_j = J_j S_z`, SVD
`J̃_j = U_j Σ_j V_{r,j}^T` with 4-D null basis `V_{n,j}`, exact projectors
`P_{R,j}=V_{r,j}V_{r,j}^T`, `P_{N,j}=V_{n,j}V_{n,j}^T` (exact SVD projectors, not the inverse
controller's damped-least-squares residual projector). Replace the uniform state cost with

```
J_z,j = ρ_R ‖P_{R,j} δz̃_j‖² + ρ_N ‖P_{N,j} δz̃_j‖²          (ρ_R < ρ_N, both nonzero)
```

equivalently `Q_j^new = S_z^{-T}(ρ_R P_{R,j} + ρ_N P_{N,j}) S_z^{-1}`, rebuilding the terminal
Riccati schedule with `Q_eff,j = Q_j^new + J_j^T Qp J_j` for consistency. `ρ_R` stays nonzero
(not zero) because `‖ξ‖` correlates strongly with `‖d̂‖` (partial corr 0.865) — fully unpenalized
task-space configuration drift risks pushing the system into worse-modelled regions.

First offline grid (current `Qp,R,Rd`, disturbance model and real constraints unchanged):
`ρ_R∈{0.01,0.1,0.25}`, `ρ_N∈{0.25,1,4}`. Evaluate the same `κ,γ^F,σ(F)` framework, plus
`‖P_R δz̃‖`, `‖P_N δz̃‖`, at the same logged states used throughout this document. Success
criterion: `σ3^constrained > 0` together with lower `γ^F` at the bad ticks, without uncontrolled
posture departure — a qualitatively different signature from anything a scalar sweep produced here.

## 13. Candidate 1 result: negative, entirely offline

Implemented as a standalone offline script (no production code touched), using SVD projectors
`P_{R,j}=V_{r,j}V_{r,j}^T`, `P_{N,j}=V_{n,j}V_{n,j}^T` of the dimensionless-scaled per-sample
Jacobian, `Q_j^new = S_z^{-T}(ρ_R P_{R,j}+ρ_N P_{N,j})S_z^{-1}`, with the terminal Riccati schedule
fully rebuilt using `Q_eff,j = Q_j^new + J_j^TQ_pJ_j` per sample. Evaluated at the same 5 ticks
(50, 125, 135, 190, 210) across the full `ρ_R∈{0.01,0.1,0.25} × ρ_N∈{0.25,1,4}` grid (9 combinations),
both truly-unconstrained and under the real 0.06/Δu/state constraints — **no robot motion involved;
this and every offline result since §3 reuses the single validated full-path live run
(`fullpath_ltv_20260917T122852Z`) as its only source of logged states.**

Result: `σ3^real ≤ 0.0001` at **every** combination, every tick — no better than uniform `Q`.
`n_active` (14–85) is essentially unchanged in magnitude across the grid regardless of how the same
total cost is split between `ρ_R` and `ρ_N`. The unconstrained response got *more* pathological, not
less: `κ_free` strongly negative (−0.51 to −0.96) and `γ^F_free` up to 1.5–1.95 everywhere in the
grid, worse than the original uniform-`Q` unconstrained response. One informative side-finding: the
*actual* current configuration departure decomposes as `‖P_Rδz̃‖=3.3–7.1` vs `‖P_Nδz̃‖=0.5–1.6` at
these ticks — real drift on triangle is predominantly task-producing, not wasteful posture drift —
but this alone wasn't sufficient to fix `σ3` at the tested weight magnitudes.

**Conclusion**: changing *what* `Q` penalizes (task-producing vs. posture) does not recover
constrained feedback authority; `n_active` barely responds to the split. This points away from any
`Q`-geometry fix and toward the *constraint* structure itself — consistent with the earlier finding
that relaxing only `|u|` didn't restore `F`, but removing every constraint (§7) did. Decision: do not
extend this grid to more extreme `(ρ_R,ρ_N)` (would only stress-test a mechanism already shown not to
be the lever, while producing an even more pathological free controller). Move to constraint-family
attribution instead — §14.

## 14. Constraint-family attribution — the joint effect, not either bound alone

Offline factorial at the same 5 ticks, original uniform `Q`, `2×2×2` relaxation of the three
constraint families (`|u|`, `|Δu|`, `z`-box) to effectively unconstrained (±1e6) one at a time and in
combination, using KKT multipliers `Λ_u,Λ_{Δu},Λ_z` (summed `|λ_i|` per family from the live OSQP
dual) alongside `κ,γ^F,σ(F)`.

- **`z`-box never binds**: `Λ_z=0` at every tick, every condition. `C_no_z` is bit-identical to
  baseline. The state box plays no role in this failure.
- **Removing `|u|` alone (`A_no_u`, `Δu` and `z` still enforced): `σ3` stays exactly `0.0000` at
  every tick.** `κ` improves partially at some ticks (0.005→0.043–0.055) but the dead direction
  persists.
- **Removing `|Δu|` alone (`B_no_du`, `u` and `z` still enforced): `σ3` also stays exactly `0.0000`
  at every tick.** Similar partial `κ` improvement, same persistent dead direction.
- **Removing both `|u|` and `|Δu|` together (`D_no_u_du`, `z` still enforced — irrelevant since `z`
  never binds anyway) reproduces the truly-unconstrained result from §7 exactly**: `σ3` revives to
  `0.13–0.20`, `κ` goes negative, `γ^F` exceeds 1.

**Neither constraint family alone is the culprit — the rank-loss signature requires both to be
simultaneously active across the horizon.** This falsifies the natural "it's the rate bound, since
`U=∞` alone didn't fix it" hypothesis in its simple form: removing only `Δu` (leaving `u` in place)
is *just as dead* as removing only `u`. Some combination of horizon-wide rows from both families is
jointly responsible; no single-family fix (e.g. simply loosening `a_max`) is expected to work on its
own, and this should be verified before any live test of a relaxed rate bound.

## 15. Reference velocity/rate headroom — ruled out as the driver

Computed directly from the planner's own `u_ref` (no MPC involved), across all 224 samples: worst-case
per-joint velocity headroom `h=0.06-|u_ref,i|` has **minimum 0.033 rad/s, median 0.047 rad/s**, and
**0% of samples fall below 0.01 rad/s**. Rate headroom `h_Δu=0.04-|Δu_ref,i|` has minimum
0.016 rad/tick, median 0.040 rad/tick, **0% below 0.005 rad**. `‖u_ref[:6]‖` never exceeds ~0.053
rad/s anywhere on the path, including at the bad-tick region (idx 121–141, 181–221).

**The feedforward trajectory itself uses only a small fraction of the available velocity/rate
budget everywhere on this path.** This rules out "insufficient feedback headroom in the
time-parameterized trajectory" as the mechanism — the reference is gentle; saturation comes almost
entirely from the feedback correction (driven by the 5–12mm tip error / `Qp`) demanding more combined
horizon-wide authority than 0.06 rad/s and its rate bound allow, not from the planned path eating the
budget. **Time-scaling the trajectory (slowing it down) would not address this** — it would shrink an
already-small `u_ref` further without touching why the feedback component saturates so heavily; not
recommended as the next offline test on this evidence.

## 16. Constrained task-only oracle — the decisive test, and the diagnosis reverses

`σ3(F)=0` under real constraints has two possible readings: "MPC is choosing the wrong command," or
"MPC is already at the feasible boundary, so more error cannot change `u0` regardless of which
controller is used." These are distinguishable by comparing the real MPC against a second QP solved
at the *same* logged states, under the *identical* real `|u|,|Δu|,z` constraints, but with the
objective stripped to task tracking only (`min_U Σ‖p̂_j-p_des‖²_Qp + ε‖U‖²`, `ε=1e-6` for
uniqueness) — a constrained task-only oracle.

| tick | error (mm) | `η_alloc = a_MPC/a_oracle` | `D_MPC/D_oracle` | `cos(u0_MPC,u0_oracle)` |
|---|---|---|---|---|
| 50 | 5.45 | 0.831 | 0.842 | 0.425 |
| 125 | 4.58 | 0.997 | 0.997 | 0.9994 |
| 135 | 5.37 | 0.996 | 0.996 | 0.9987 |
| 190 | 11.20 | 0.966 | 0.967 | 0.944 |
| 210 | 10.84 | 0.900 | 0.900 | 0.926 |

**`η_alloc` is close to 1 at every tick tested**, and at the two ticks with the largest command norms
(125, 135) `u0_MPC` and `u0_oracle` are nearly identical (`‖Δu0‖=0.005–0.007` on commands of norm
~0.09–0.10). **MPC is already allocating its constrained authority almost as efficiently as a
controller that cares about nothing but tip tracking.** `σ3(F)=0` here reflects saturation, not a
poor cost/horizon choice.

Also notable: at ticks 125 and 135, *both* controllers — including the oracle — produce a **negative**
`D_k` (`-10.3`, `-12.2`) and negative `a_k`: even the best feasible one-step command increases
near-term error. This is expected and not a defect — a proper receding-horizon optimum can accept
near-term error growth to set up a better trajectory given the `Δu` redirection cost over the
remaining horizon; since even the myopic task-only oracle can't do better in one step, this is a real,
controller-independent physical limitation at those ticks.

**Verdict on the decision framework from §14/§15's follow-up**: `η_alloc≈1` closes the "poor
constrained choice vs. already-optimal" question decisively in favor of *already near-optimal*.
Per that framework: stop tuning the cost entirely (any further `(Q,Qp,R,Rd)` search is expected to be
close to exhausted, consistent with every scalar/geometric sweep in §10/§13 already failing to move
`σ3^real`). The active-constraint-chain attribution and stagewise release map (which were aimed at
diagnosing *why* a poor allocation happens) are not needed given this result. **The next Stage-A
direction is adaptive path-progress/reference governance** — introduce a progress state `s` and allow
`ṡ` to decrease when feedback authority is exhausted (rather than forcing the reference to keep
advancing through saturation), instead of any further cost- or constraint-weight search.

## 17. Candidate 2 (adaptive path-progress governor) — rejected in its simple form

Offline governed-horizon sweep at the same 5 ticks, `α∈{1,0.75,0.5,0.25,0}`: fractional path-progress
`s_{j|k}=k+jα`, linear interpolation of `z_ref,u_ref,p_des,J` at fractional sample index, `u_ref^(α)=
α·u_ref`, same `Q,R,Rd,Qp`, same real `u,Δu,z` constraints, `d̂_k` held constant across the horizon
(unchanged model). Multi-step recovery metric `Γ_H = Σ_{j=1}^{5}‖ê_{j|k}‖² / (‖e_k‖²·5)`, `h=5`
(0.5s). Terminal DARE cost omitted for this diagnostic (justified by §8's finding that it has no
first-order effect on `u0` in the unmodified problem).

| tick | `Γ_H(α=1)` | `Γ_H(α=0)` | direction |
|---|---|---|---|
| 50 | 0.647 | 0.635 | marginal improvement |
| 125 | 0.579 | 0.620 | **worse** |
| 135 | 0.441 | 0.479 | **worse** |
| 190 | 0.923 | 0.900 | marginal improvement |
| 210 | 1.004 | 0.987 | marginal improvement, barely crosses 1 |

Predicted horizon RMS moves at most ~1–2% even at a full hold (`α=0`) — e.g. tick 190:
10.76mm→10.62mm. `n_active^u` does drop meaningfully with `α` (tick 210: 47→26 across the sweep),
so the governor genuinely relieves constraint saturation — but that relief does not translate into
materially better predicted tracking, and at 2 of 5 ticks slowing is outright counterproductive.

**Diagnosis**: `d̂_k` (5–12mm at these ticks, the same order of magnitude as the entire tracking
error) is held constant across the horizon in the current model regardless of `α` — it is a snapshot
of the current nonlinear model-plant mismatch, not a function of path speed. The governor only removes
the "target is outrunning the controller" component of error; the dominant persistent-residual
component (already characterized as slowly-varying via `C_d(15)≈0.90`, §prior sessions) is untouched
by slowing progress. **Candidate 2 changes future reference demand, but not the dominant persistent
model-plant residual — it attacks the wrong term.**

**Rejected as recorded**: adaptive path progress substantially reduced constraint activity but did
not materially improve predicted recovery. Not rescued by making `d̂` depend on `α` — that would
require evidence for a model like `d_{j|k}=d(s_{j|k})`, which is exactly the next question (§18), not
an ad hoc fix to save the governor. No further governor elaboration attempted.

## 18. Candidate 3 groundwork: is `d_phys` repeatable enough to preview?

Before building any residual-preview controller, testing the precondition: does the same triangle
path produce a *similar* nonlinear model-plant residual `d_phys(s)` across independent runs? If not,
a path-indexed lookup `d̄(s)` is not viable and the next investigation should target the physical
hidden state (hysteresis, beam dynamics, insertion error, calibration) instead.

Computed `d_phys` purely from geometry (`q_meas_rad`, `tip_mm`, current forward beam model — no QP
replay, unaffected by the §2 stale-plan issue) for three independent triangle runs, aligned by
`ref_index`: **A** = 2026-09-16 `mpc_ltv`+FF; **B** = 2026-09-17 `mpc_ltv_offline` direct-apply (the
run this whole document is built on); **C** = 2026-09-17 `naive_inverse_jacobian_ltv`, kp=1.0.

| pair | same day? | same controller family? | `C_ab` | `E_d,ab` |
|---|---|---|---|---|
| A vs B | No | similar (both MPC) | +0.279 | 7.98mm |
| A vs C | No | No | **+0.818** | 4.62mm |
| B vs C | **Yes** | No | **−0.230** | 11.28mm |

**Result is mixed, and the pattern argues against a simple path-indexed model.** The strongest
correlation is between the two *most different* runs (different day, different controller family);
the same-day pair — which should be easiest for a repeatable path-indexed bias, since calibration and
environment are most similar — is negative. Likely explanation: A (FF) and C (inverse-Jacobian) both
track close to the planned joint trajectory, so they visit similar configurations at each path index;
B (direct-apply MPC, subject to the saturation this document diagnoses) visits substantially different
configurations at the same nominal index. Combined with the already-established `‖ξ‖`↔`‖d̂‖` partial
correlation (0.865, established in earlier sessions), this suggests `d_phys` is better modeled as a
function of the actual visited **configuration** `z` (or the redundant/null coordinate `ξ`), not of
path progress `s` alone.

**Precondition for Candidate 3 as literally specified (`d̄(s)` path-indexed lookup) fails.** Not a
rejection of anticipatory disturbance compensation in general — it redirects the idea toward a
configuration-indexed model `d(z)` or `d(ξ)`, tested next.

## 19. Configuration-conditioned repeatability — the vector-level test

Cross-run only (all pairs between different runs, same 3 runs A/B/C as §18), computing normalized
distances `d_z=‖S_z^{-1}(z_a-z_b)‖`, `d_ξ=‖S_z^{-1}ξ_a - S_z^{-1}ξ_b‖`, residual difference
`E_d=‖d_phys,a-d_phys,b‖`, and residual cosine `C_d`. This tests the *vector* residual, a materially
stronger requirement than the earlier scalar `corr(‖ξ‖,‖d̂‖)=0.865`.

- **`d(z)` (absolute configuration): no signal.** Pooled `corr(dz,Ed)=+0.066`, `corr(dz,Cd)=-0.027`
  (noise level). Decile-binned `E_d` stays flat at 9.7–10.6mm and `C_d` flat around `-0.05` to
  `-0.24` across the *entire* distance range, non-monotonic.
- **`d(ξ)` (departure from reference): a real magnitude trend, but weak direction.** `E_d` rises
  fairly monotonically from 6.44mm (closest decile) to 12.74mm (farthest) — the correct-sign
  confirmation of the known scalar relationship. But `C_d` is already near zero at the *closest*
  decile (`-0.071`, not strongly positive as genuine vector predictability requires), only drifting
  to `-0.49` at the far end. **Even the most configuration-matched cross-run pairs don't have
  well-aligned residual vectors** — the `E_d`-vs-`d_ξ` trend looks like it is largely inherited from
  the already-known magnitude relationship, not genuine cross-run reproducibility of the residual's
  *direction*.

## 20. Predictive-model test (k-NN, leave-one-run-out) — fails on the run that matters

`M_s=d̄(s)` (path-indexed lookup, §18), `M_z`, `M_ξ` (k-NN, k=7, normalized coordinates), against a
`d̂=0` baseline, each trained on two runs and evaluated on the third:

| held out | `d̂=0` | `d̄(s)` | `M_z` | `M_ξ` |
|---|---|---|---|---|
| A | `R²=-0.03` | `R²=0.77` | `R²=-0.27` | `R²=0.37` |
| **B (direct-apply MPC — the actual failure regime)** | **`R²=0.45`** | `R²=-0.29` | `R²=-0.08` | `R²=0.25` |
| C | `R²=0.30` | `R²=0.27` | `R²=0.37` | `R²=-0.75` |

No model wins consistently across all three held-out runs. Critically, **on run B — the direct-apply
MPC run this entire document is about — the trivial zero-prediction baseline beats every learned
model**, and `d̄(s)` is catastrophically negative there (`-0.29`). A's strong `d̄(s)` result is very
likely an artifact of C (in its training pool) already correlating strongly with A per §18's pairwise
test (+0.818), not evidence of a genuine repeatable physical model.

**Decision rule from §18 applied**: fails condition 3 ("the prediction remains useful on the
direct-apply MPC run B, not just the tightly tracking runs A/C") decisively, and specifically where
it matters most. Combined with §19's near-zero vector alignment even at the closest configuration
matches, **the evidence points away from a memoryless mapping `d=f(z)` or `f(ξ)`, toward missing
hidden state** (hysteresis, beam dynamics, insertion-history effects, calibration drift) rather than
any function of the instantaneous configuration alone. Candidate 3 is not implemented. This is the
recorded stopping point for the constrained-feedback investigation opened by §1 — the next
investigation, if pursued, is qualitatively different (identifying hidden/dynamic state), not another
memoryless residual model or controller-cost redesign.

## 21. Session close — narrowed frozen conclusion and forward plan

**Narrowed conclusion** (tighter than "constraint redesign is ruled out"): *given the current
physical constraints, the direct-apply MPC is already allocating feasible authority nearly optimally*
(§16, `η_alloc≈0.83-1.00`). `Q, Qp, R, Rd`, the terminal cost, task/null weighting, and simple path
progress are not where the evidence points anymore (§10, §13, §17). The open question has moved from
"how should MPC choose a better command" to a plant-model question: `p_tip ≠ f(z)` alone. §19-§20's
cross-run failure of both `d(z)` and `d(ξ)` shows some *additional* state matters.

**Do not revisit** (tested enough to justify moving on): `Q/Qp/R/Rd` sweeps, null/posture weighting,
terminal-cost redesign, simple `α`-governors, path-indexed disturbance lookup, memoryless `d(z)`
learning.

**Prioritized hidden-state candidates**: motion/history state, beam hysteresis/relaxation, insertion
state error, magnetic history, run-dependent calibration drift. Insertion is flagged as especially
high-priority: §4 already established `L_measured` is not actually measured — `insertion_length_m` is
pure software bookkeeping (dead-reckoned from `cfg.initial_insertion_m`), decoupled from whatever the
physical advancer actually executes. The beam-model state can be wrong even when the controller
believes it knows `L`.

**Next-session plan** (not started — explicitly deferred past today's stopping point):

1. **Offline history/ARX test.** Model `Δd_{k+1}=d_{k+1}-d_k` (not `d_k=f(z_k)`, which has failed)
   from features `d_k, d_{k-1}, Δz_k, u_k, u_{k-1}, Δu_k`, and insertion-specific terms
   (`L_k^software`, `u_{L,k}^cmd`, a short trailing sum of `u_{L,τ}^cmd·Δt`). Start with a regularized
   linear ARX model `d_{k+1}=A d_k+B_1Δz_k+B_2u_k+B_3Δu_k+c`, not a high-capacity nonlinear model.
   Critical baseline: **persistence** `d̂_{k+j|k}=d_k` (what the current controller already assumes),
   not `d̂=0` — the dynamic model must beat persistence, evaluated at `h=1,5,15` steps ahead,
   leave-one-run-out (same A/B/C convention as §18-§20), reporting RMSE and vector cosine for both.
2. **Live same-configuration/different-history hold experiment** (the stronger test). 3 triangle
   configurations (benign / medium-residual / severe 5-12mm region), each approached from two
   directions (`z*-Δz→z*` and `z*+Δz→z*`), held 5-10s, recording `q_RTDE(t), p_tip(t), L_software(t),
   u_L^cmd(t), d_phys(t)`. Diagnostic outcomes: converging steady residuals with different transient
   decay ⇒ missing relaxation dynamics; different steady residuals at nominally the same state ⇒
   genuine hysteresis; no relationship to history at all ⇒ look harder at calibration/insertion drift.
3. **Insertion-state instrumentation.** Treat `L_software` as an estimate, not a measurement — if any
   independent encoder/step-count/end-stop reference can be exposed (even temporarily, for system ID),
   directly measure `L_physical - L_software` and whether it accumulates with motion
   direction/history. A few tenths of a millimetre of insertion error could plausibly explain the
   poor cross-run `d(z)` repeatability, residual persistence, and controller-dependent residual
   patterns observed in §18-§20.

The new question going forward: **what state variable, absent from `z=[q1,...,q6,L]`, explains the
model-plant residual?**
