# Controller justification: delay-aware task-nullspace MPC (γ=0, R700)

**Status:** offline phase complete, frozen for first live validation (2026-09-20).
**Scope:** every term in the objective function, the prediction horizon, the
delay/execution model, and the hard constraints of the controller shipped in
`run_mpc_delay_aware_nullq_r700.py`. Each subsection states the frozen value,
the competing alternatives that were actually tested, and the quantitative
evidence that decided between them. Nothing here is asserted without a
replay, unit test, or live run behind it; where a claim rests on work from an
earlier phase of this project (not reproduced in this document's own tables),
that is stated explicitly with its source.

All offline numbers in this document come from replaying candidate
controllers against the **same fixed 268-tick log** (the successful live run
`close_loop_logs/myrun/mpc_delay_aware_accumC_20260918T165039Z`, plan
`plans/rectangle_10x15mm_skipglobal_2026-09-12`), so every table is a
same-inputs, different-controller comparison — never a different run, camera
session, or robot state confounding the result.

---

## 1. Objective function, as shipped

$$
\mathcal J=\sum_{j=1}^{15}\Big[\;\underbrace{\|P_{N,j}S_z^{-1}(z^{phys}_{j|k}-z_{\rm ref,k+j})\|^2}_{Q_N}
\;+\;\underbrace{\|\hat p_{j|k}-p_{\rm des,k+j}\|_{Q_p}^2}_{Q_p}
\;+\;\underbrace{\|u_{j|k}-u_{\rm ref,k+j}\|_{R_{700}}^2}_{R}
\;+\;\underbrace{\|\Delta u_{j|k}\|_{R_d}^2}_{R_d}\;\Big]
$$

subject to the delay-aware prediction model (§2) and the hard constraints
(§6). $V_f=0$ — no terminal cost. Every symbol here is justified in its own
section below.

---

## 2. Delay/execution model: $d=2$, undelayed insertion, execution-C accumulator

**Frozen:**

$$
c_{j+1|k}=c_{j|k}+\Delta t\,u_{q,j|k},\qquad
q^{phys}_{1|k}=c_{k-1},\quad q^{phys}_{2|k}=c_k,\quad q^{phys}_{3|k}=c_k+\Delta t\,u_{q,0|k}
$$
$$
L_{j|k}=L_k+\Delta t\sum_{r=0}^{j-1}u_{L,r|k}\quad(\text{insertion: undelayed})
$$

**Evidence (established in an earlier phase of this project; re-extracted
and independently recomputed from the saved fold data as part of this
freeze — see `SUPPLEMENTARY_EVIDENCE.md` §S10 for the full per-fold table):**

- `prediction.py`'s module docstring cites the physical identification: **5-fold
  leave-one-out cross-validation, $G_{\rm delay}=0.945\text{–}0.986$**, for the
  $d=2$ joint-command delay (`hardware/online/rectangle_stage_a/README.md`).
  Re-deriving $G$ directly from the saved fold RMSEs (§S10.1) gives
  **0.973–0.992** across the 5 held-out folds — consistent with, though not
  numerically identical to, the docstring's figure (likely a slightly
  different $G$ formula) — both agree the $d=2$ model's held-out error is
  roughly **6–11× smaller** than a no-delay model's on every fold. A
  fractional-delay alternative ($\tau\approx2.16$ samples) was also fit per
  fold and found **worse than $d=2$ on 4 of 5 held-out folds** (§S10.2) —
  the basis for freezing the simpler integer $d=2$ model rather than a
  fitted fractional delay. A command issued at tick $k$ first physically
  affects the joints at $k+3$, not $k+1$ — validated by unit-tested indexing
  (`prediction.py`'s own self-test), not assumed.
- Insertion (`L`) was identified separately as **undelayed** — $d=2$ applies
  only to the six UR joints, not the advancer (see
  `inverse_jacobian_2dof_delay_aware.py`'s module note, referenced from
  `prediction.py`).
- `delay_samples` is hard-capped at 2 in `build_delay_prediction_matrices`
  (raises for any other value) specifically because $z_0^{aug}$ carries only
  one step of command history — this is a deliberate scope limit, not an
  oversight, and it is exactly the amount of history the identified delay
  needs.
- **Execution-C accumulator + 50Hz interpolated `servoJ` streaming**: prior to
  this delay-aware work, a direct-apply command seam was measured to realize
  only **~20% of commanded joint increments** ($G=0.199$); switching to a
  controller-owned persistent accumulator raised realization to $G=1.000$,
  and adding 50Hz streaming on top of the 10Hz outer tick additionally
  dropped mean/max command-tracking error **~4×** (11.8→2.7 mrad mean,
  36.2→8.7 mrad max). This is why `cfg.accumulator_seam=True` and
  `servo_stream_hz=50` are frozen, non-negotiable parts of the execution
  layer for every controller variant compared in this document — without
  them, none of the cost-function comparisons below would be measuring the
  controller rather than the actuation seam.

**Not re-validated in this document:** the $d=2$/$G_{\rm delay}$ identification
itself predates the objective-function ablation campaign described below and
is treated as a fixed, previously-established input to it.

---

## 3. Disturbance model: $\beta_d=1$ (constant persistence across the horizon)

**Frozen:** $\hat d_{k+j|k}=\beta_d\cdot d_k$ for all $j=1,\dots,N$, i.e. the
instantaneous output residual estimated from the current measurement is
assumed to persist, undecayed, across the whole horizon.

**Evidence:** established in an earlier phase of this project (the
$\beta_d=1$ vs a decaying alternative $\beta_d=0.82$ was tested; $\beta_d=1$
was adopted as the frozen first-ablation value, with $0.82$ noted as an
"earned but not yet adopted refinement" — see `delay_aware_mpc.py`'s class
docstring). A direct sensitivity re-run (`SUPPLEMENTARY_EVIDENCE.md` §S11,
262-tick live-log replay) confirms this is not an inconsequential choice:
switching $\beta_d$ 1.0→0.82 changes the commanded solution by 16.3% RMS
and the predicted beam-tip trajectory by up to 1.45mm RMS over the horizon.
This is a **sensitivity** result (the choice matters) not a **superiority**
result (1.0 is not shown better than 0.82) — $\beta_d=1$ (constant
persistence, the conservative assumption) was kept specifically so it could
not confound the state-cost/$R$ comparisons in §6–§9, which all hold it
fixed identically across every condition. Adopting $\beta_d=0.82$ remains an
open, evidenced refinement for later, not a rejected alternative.

---

## 4. Prediction horizon: $N=15$, $\Delta t=0.1$s, $V_f=0$

**Frozen:** 15-step horizon (1.5s lookahead) at the 10Hz outer control rate,
no terminal cost.

**Evidence:**

- $N=15$ and $\Delta t=0.1$s were inherited unchanged from the
  already-validated undelayed MPC baseline (M0's own predecessor) and kept
  fixed across **every single condition** in this campaign (MT, $Q_\gamma$
  sweep, $R$/$R_d$ sweeps, insertion ablations, $Q_N$ ablation) specifically
  so that no result in this document could be explained by a horizon change
  rather than a cost-function change.
- $V_f=0$ (no DARE terminal cost) is a hard requirement of
  `DelayAwareBeamOutputTrackingMPC.__init__`, which **raises** if
  `beam_config.use_dare_terminal_cost` is true — the class's own docstring
  states the terminal Riccati machinery was never derived for the augmented
  13D delay state, so silently reinterpreting it would be worse than doing
  without it. This is a scope decision, not a tuned-away term: no ablation
  compared $V_f=0$ against a working terminal cost, because a working one
  does not yet exist for this state augmentation.
- $N=15$ itself was swept (on the predecessor, non-delay-augmented LTV MPC —
  see the caveat in `SUPPLEMENTARY_EVIDENCE.md` §S12): RMS tracking error is
  essentially converged by $N=5$ and **bit-identical from $N=10$ to $N=15$**
  on a fixed reference trajectory (§S12.1), and independently on **both** the
  rectangle and triangle plan shapes (§S12.2), where $N=1$ is additionally
  shown to be qualitatively worse — a negative median correction-direction
  cosine (the controller fights itself tick to tick) that turns positive and
  stable from $N=3$ on. $N=15$ therefore sits deep inside the converged
  plateau, not at a marginal or arbitrary cutoff, while solve time at $N=15$
  (≈5.8ms in that sweep; 8.1ms for the actual frozen nullspace/R700 solve,
  §S1.3) remains far below the 70–100ms control budget. This sweep predates
  the delay-augmented formulation, so it establishes the horizon-convergence
  *shape* for this beam/robot system rather than re-validating $N=15$
  specifically for the 13D delay-augmented state.

---

## 5. Task cost $Q_p$: unchanged throughout, $Q_p = 1.0\cdot\mathrm{diag}(1/(0.5\,\mathrm{mm})^2)$

**Frozen:** isotropic, in the same 2D beam-plane projection $C$ used
everywhere else in this document (`build_beam_plane_projection`, axial axis
$(-1,0,0)_R$, normal axis $(0,0,-1)_R$ in the robot frame), weight
unchanged from the pre-existing validated undelayed/delay-aware MPC.

**Evidence — this weight was never swept in this campaign, and that is
itself justified by a standing result:** across every condition tested (M0,
MT-stagewise, MT-constant, $Q_\gamma\in\{1,0.25,0.10,0\}$, the full $R/R_d$
sweeps, both insertion ablations), the **prediction-accuracy metric**

$$
E_{\rm model,+j}=\mathrm{RMS}\|p_{\rm meas,k+j}-\hat p_{k+j|k}\|
$$

stayed in the **0.34–0.6mm range** throughout — i.e. the beam-tip model
itself was never the limiting factor; every ablation in this document is
legitimately a **$E_{\rm chosen}$** (trajectory-choice) question, not an
$E_{\rm model}$ (model-fidelity) question, which is precisely the
decomposition this campaign relied on to interpret every subsequent result
correctly. Because $Q_p$'s corresponding cost-gradient term
($G^TQ_p\bar G$) was never implicated by that decomposition as a source of
poor tracking, it was left untouched rather than swept.

---

## 6. State cost: from isotropic $Q$ (M0) to exact task-nullspace $Q_N$ (γ=0)

This is the single structural change this document exists to justify. Three
competing formulations were built, unit-tested, and replayed before
converging on the frozen choice.

### 6.1 Formulation A (rejected): target-consistent shift "MT"

Shift only the **target** the isotropic $Q$ tracks — $z_{\rm ref}\to z^t$,
solved via weighted-min-norm inverse in the 2D task subspace — keeping $Q$
itself isotropic. Two variants: **MT-stagewise** (independent target per
horizon stage) and **MT-constant** (one offset computed at the first
joint-authority stage, held fixed across the horizon).

**Evidence against:**

| condition | note |
|---|---|
| MT-stagewise vs MT-constant | nearly identical behavior — disproves the "the stagewise target is dynamically inconsistent" hypothesis that motivated testing MT-constant in the first place |
| Gradient/KKT decomposition | none of the pre-declared go/no-go criteria for live testing were cleanly met (task/nullspace gradient antagonism was not substantially reduced vs M0; MT was not at least as cost-efficient as brute-force $Q_p\times 4$) |
| Offline replay | $E_{\rm chosen}$ improved, but at the cost of substantially higher constraint saturation than M0 |

**Verdict:** MT rejected outright as a live candidate — a structurally
different formulation ($Q_\gamma$, §6.2) was pursued instead, not a
refinement of MT.

### 6.2 Formulation B: task/nullspace $Q_\gamma$, sweep over γ

$$
\ell_{Q,j}=\tilde z_j^T(P_{N,j}+\gamma P_{R,j})\tilde z_j,\qquad
\tilde z=S_z^{-1}(z-z_{\rm ref})
$$

with $P_{R,j},P_{N,j}$ **exact orthogonal projectors** built from the
numerical-rank SVD of the stagewise scaled task Jacobian
$J_{s,j}=(C J_j)S_z$ (rank confirmed $=2$ at every stage — a 2D in-plane
task — not hard-coded). $\gamma=1$ recovers M0's isotropic $Q$ exactly (unit
test: $H$ diff $1.86\times10^{-9}$, $f$ diff $0.0$, $u_0$ diff
$4\text{–}7\times10^{-16}$ — this equivalence is kept in CI permanently as a
regression guard, see §11).

**6-condition sweep** (M0, M0+4$Q_p$, MT-stagewise, MT-constant,
$Q_{\gamma=0.25}$, $Q_{\gamma=0.10}$), goal: Pareto improvement in
$E_{\rm chosen}$ vs actuator saturation, not minimum $E_{\rm chosen}$ alone.
**$\gamma=0.10$ was the standout** on stagewise constraint-activity decay.

**$R/R_d$ isolation at fixed $\gamma=0.10$** ($R\times\{3,10\}$,
$R_d\times\{3,10\}$ independently): only mild effects. A real
implementation bug was found and fixed during this sweep —
`effective_input_increment_scale` silently auto-derives from
`acceleration_limit` when `input_increment_scale` is left `None`, making
$R_d$'s FREE-BOTH-diagnostic normalization **~2500× looser** than intended
and making the $R_d\times3/10$ sweep come back bit-identical to baseline
regardless of the requested multiplier. After the fix, $R\times\{3,10\}$ at
$\gamma=0.10$ still showed only mild traction — **this negative result is
what motivated testing $\gamma=0$ next**, on the reasoning that a residual
row-space weight of 0.10 was still large enough to dominate $R$'s effect.

**Verdict:** $\gamma=0.10$ retained as a valid, better-than-M0 negative
result, but superseded by $\gamma=0$ once $R$ was shown to gain real
traction there (§6.3) — not adopted as final.

### 6.3 Formulation C (frozen): exact task-nullspace only, $\gamma=0$

Removing the row-space weight *entirely* (not just to 0.10) is a
structurally different claim than an MT-style target shift: it is provably
true that once $Q$ is nullspace-only,

$$
P_N(\tilde z-\delta\tilde z^t)=P_N\tilde z\quad\text{for any task-producing }\delta z^t
$$

i.e. MT's whole shifting mechanism becomes a no-op — this is why $\gamma=0$
is a genuinely different structural change, not a refinement chain
MT→$Q_{0.10}$→$\gamma{=}0$.

**Unit tests** (`stagewise_task_nullspace_self_test.py`, 4/4 pass, re-run
2026-09-20 against the live schedule as part of this freeze):

| test | result |
|---|---|
| Projector identities ($P_N^2=P_N$, $P_N^T=P_N$, $P_RP_N\approx0$, $P_R+P_N=I$) at every stage | PASS, every stage |
| Pure row/null vector annihilation ($J_s\cdot(P_N x)\approx0$) | PASS |
| Numerical rank $=2$ at every stage (computed, not assumed) | PASS |
| $\gamma=1$ reproduces M0 exactly | PASS — $H$ diff $1.86\times10^{-9}$, $f$ diff $0.0$, $u_0$ diff $6.94\times10^{-17}$ |

**Logarithmic $R$ scale-finding at $\gamma=0$ fixed** ($R\times\{1,10,100,1000\}$,
$R_{d,0}$ unchanged): $R1000$ reached $\rho_{\rm desire}=0.661$ with
$E_{c,+10}=0.298$mm, vs M0's $0.755$mm.

**$R$ refinement** ($R\times\{300,500,700\}$): $R700$ and $R1000$ gave
essentially identical joint-channel activity ($P(\Delta u_{q}\text{ active})$
$0.037$ vs $0.041$) — $R700$ is not unnecessarily conservative relative to
$R1000$.

**The major mid-campaign discovery, decomposed by actuator (R700, γ=0):**

| actuator | $P(\cdot\text{ active})$ |
|---|---|
| joints $q_2$–$q_6$ | **exactly 0.000**, in *every single condition tested all day* (M0, MT, $\gamma$-sweep, entire $R$-sweep) |
| insertion $L$ | 0.556 |

i.e. "constraint activity" as measured all campaign long was almost entirely
an insertion-channel phenomenon, not joint aggressiveness — this
reclassified every subsequent saturation metric and directly motivated the
insertion-vs-joint authority investigation below.

**Axis-resolved $E_{\rm chosen}$, M0 vs R700/γ=0 (the decisive comparison):**

| | $E_{c,x,+3}$ | $E_{c,x,+5}$ | $E_{c,x,+10}$ | $E_{c,y,+3}$ | $E_{c,y,+5}$ | $E_{c,y,+10}$ |
|---|---|---|---|---|---|---|
| M0 | 0.419 | 0.425 | 0.444 | 0.593 | 0.592 | 0.607 |
| R700, γ=0 | 0.264 | 0.201 | **0.094** | 0.486 | 0.335 | **0.238** |

M0's lateral ($y$) error is essentially flat across the whole horizon —
it never converges. R700/γ=0 drops predicted $y$-error 61% (0.607→0.238mm)
and $x$-error 79% (0.444→0.094mm, well below M0's level) — this is the axis
where the live kp=1 inverse controller had its advantage over M0, so this is
the result that specifically justifies expecting a live improvement, not
just an aggregate-RMS improvement that could be hiding an uneven tradeoff.

**Task-direction (physical division-of-labor) decomposition, R700/γ=0:**

| feedback source | mean\|x\| | mean\|y\| | frac\|x\| | frac\|y\| |
|---|---|---|---|---|
| $\Delta p_q$ (joint feedback) | 0.030mm | 0.123mm | 0.212 | **0.955** |
| $\Delta p_L$ (insertion feedback) | 0.069mm | 0.018mm | **0.905** | 0.278 |

Insertion feedback is axial ($x$)-dominant, joint feedback is lateral
($y$)-dominant — matching the physical roles: insertion advances the beam,
joints steer it magnetically. This shows the optimizer's actuator
allocation is not an artifact of miscalibrated weights; it is exploiting
real physical structure, exactly as the kp=1 inverse controller's own
(disjoint) actuator allocation already does (§9).

**Verdict:** $\gamma=0$ (exact task-nullspace $Q_N$) is the frozen state-cost
formulation.

### 6.4 Is $Q_N$ itself necessary, given full $u_{\rm ref}$ already anchors posture? — tested and answered: yes

**Hypothesis tested:** the kp=1 inverse controller has zero nullspace
feedback correction ($k_n=0$) but still tracks the planner's own nullspare
trajectory, because it advances along $u_{\rm ref}$'s own redundant motion.
Maybe $Q_N$ is therefore redundant for the MPC too, once $u_{\rm ref}$ is
kept full.

**Ablation:** A ($Q_N$ on, R700, full $u_{\rm ref}$) vs B ($Q_N=0$
entirely — not just $\gamma=0$'s row-space term, the *whole* state cost —
R700, full $u_{\rm ref}$ unchanged), replayed on the same 268-tick log,
comparing task tracking, actuator effort, and accumulated nullspace
deviation $e_N(k)=P_{N,j}S_z^{-1}(z_{j|k}-z_{\rm ref,k+j})$ using each
condition's own predicted physical-state trajectory.

| metric | A ($Q_N$ on) | B ($Q_N=0$) |
|---|---|---|
| $E_{c,tot,+3}$ / $+5$ / $+10$ (mm) | 0.557 / 0.404 / 0.279 | 0.552 / 0.387 / 0.248 |
| RMS joints (mrad) | 20.98 | 21.69 |
| $P(u_L\text{ active})$ | 0.556 | 0.552 |
| RMS$\|e_N\|$ at stage +3 | 0.174 | 0.208 |
| RMS$\|e_N\|$ at stage +10 | 0.276 | **0.524** (1.9×) |
| RMS$\|e_N\|$ at stage +15 (terminal) | 0.290 | **0.695** (2.4×) |
| max$\|e_N\|$ at stage +15 | 0.762 | **1.888** (2.5×) |

Task tracking and actuator effort are essentially unchanged with $Q_N$
removed — but predicted nullspace configuration drift roughly **doubles by
mid-horizon and grows to 2.4–2.5× by the terminal stage**. This is exactly
the signature that justifies keeping $Q_N$: it costs nothing in task
tracking on this trajectory, but it is doing real work holding the
horizon-predicted posture near the planner's redundant trajectory that full
$u_{\rm ref}$ alone does not fully provide.

**Verdict:** $Q_N$ retained. The minimalist "$Q_p+R(u-u_{\rm ref})+R_d$"
form (no $Q_N$ at all) was tested directly and rejected on this evidence.

---

## 7. Input tracking cost $R$: $R_{700}=700\,R_0$, $R_0=10^{-2}\mathrm{diag}(1/s_u^2)$

**Frozen:** global scalar `input_tracking_weight = 7.0` (i.e. $700\times$
the original $10^{-2}$), same per-channel scales as always
($s_{u,q}=0.05$rad/s, $s_{u,L}=5$mm/s — **not** rescaled, see §9).

**Evidence:** see §6.3's logarithmic sweep and refinement
($R\times\{1,10,100,1000\}$, then $\{300,500,700\}$) — this is the same
evidence, since $R$ was only ever tuned *jointly* with the $\gamma=0$
state-cost change, not independently. $R700$ was selected over $R1000$
because the two are statistically indistinguishable on joint-channel
activity ($P(\Delta u_q\text{ active})$ $0.037$ vs $0.041$) — i.e. $R700$ is
not leaving easy performance on the table relative to a stronger multiplier,
so the smaller, less-aggressive value was preferred.

**Why the channel-scale asymmetry ($s_{u,q}=0.05$ vs the hard limit
$0.1$rad/s; $s_{u,L}=5$mm/s vs the hard limit $2$mm/s) was investigated and
then left alone:** see §9 — a direct ablation of $s_{u,L}$ (5→2mm/s) showed
this asymmetry is not simply "making insertion artificially cheap"; the
joints do not step in to compensate when insertion is made more expensive,
so the asymmetry was not treated as a redistributable weighting bug.

---

## 8. Input increment cost $R_d$: $\texttt{input\_increment\_weight}=10^{-3}$ (unchanged), normalization explicitly pinned

**Frozen:** original weight, untouched. `input_increment_scale` is
**explicitly pinned** to $(\Delta t\cdot a_{\rm max})$ for every channel in
both the offline replay configs and the live worker config — never left to
auto-derive.

**Evidence:**

- The $R/R_d$ isolation experiment at $\gamma=0.10$ (§6.2) surfaced a real
  bug: `ConfigurationMPCConfig.effective_input_increment_scale` silently
  derives from `acceleration_limit` when `input_increment_scale` is `None`.
  My own FREE-BOTH diagnostic configs used a relaxed
  `acceleration_limit=1e3` to open up the hard velocity/rate bounds for a
  $\rho_{\rm desire}$-style unconstrained-optimum diagnostic — which
  silently made $R_d$'s **cost normalization** ~2500× looser too, producing
  a false negative ($R_d\times3/10$ came back bit-identical to baseline).
  Fixed by pinning `input_increment_scale` explicitly on every subsequent
  config. This is why the live worker (`run_mpc_delay_aware_nullq_r700.py`)
  computes and passes `input_increment_scale` explicitly rather than relying
  on the field's default derivation, even though in the live (non-relaxed)
  context the auto-derived value happens to coincide with the correct one —
  pinning it removes that coincidence as a future failure mode.
- Joint rate activity was found to already be small at $R700$
  ($P(\Delta u_q\text{ active})=0.037$), and essentially all remaining
  "activity" headline numbers were shown (§6.3) to be an insertion-channel
  phenomenon, not a rate-limiting one — so $R_d$ was never identified by any
  diagnostic in this campaign as the binding term, and tuning it was
  explicitly deferred rather than swept broadly.

---

## 9. Insertion authority: kept at the original $s_{u,L}=5$mm/s, $|u_L|\le2$mm/s — investigated, not weakened

Insertion showed the highest constraint activity of any channel throughout
the campaign ($P(u_L\text{ active})=0.556$ at R700/γ=0, up to $94.9\%$
task-contribution $\eta_L$ at some ticks). Four separate diagnostics were
run specifically to determine whether this was a tuning artifact (fixable
by reweighting) or a genuine structural property (to be preserved) before
deciding not to touch it.

| diagnostic | result | conclusion |
|---|---|---|
| kp=1 inverse controller comparison | INV uses **zero** insertion feedback (RMS $\delta u_L=0.0000$mm/s exactly, $P(u_L\text{ active})=0.000$ exactly) yet tracks well | establishes the comparison target, not proof MPC is wrong to use insertion |
| $s_{u,L}$ rescale ablation (5→2mm/s, joints untouched) | insertion usage drops, but **joints do not compensate**; $E_{\rm chosen}$ gets slightly *worse* | rules out "insertion is simply mis-costed relative to its headroom" as the explanation |
| $\delta u_L=0$ forced (equality constraint, matching INV's own allocation exactly) | $E_{c,+10}$ **more than doubles** (0.279→0.571mm); joint effort is essentially **unchanged** (RMS 20.98→21.08mrad, $P(\Delta u_q\text{ active})$ 0.037→0.041) | joints genuinely cannot substitute for insertion in this formulation — this is not laziness, the optimizer has no cheap alternative |
| Horizon authority decomposition, $\|Q_p^{1/2}\partial\hat p_j/\partial u_{q,0}\|$ vs $\|Q_p^{1/2}\partial\hat p_j/\partial u_{L,0}\|$ | joint authority **exactly 0.0000** at stages 1–2 (structural — zero authority under $d=2$ until stage 3), rising to ~32.85–34.19 from stage 3 on; insertion authority **~200–201, constant at every stage** — roughly 6× larger than active joints | insertion is a fundamentally more task-efficient actuator here, not just earlier-acting |
| $Q_{p,1}=Q_{p,2}=0$ ablation (remove the *only* incentive insertion has that joints structurally lack) | $P(u_L\text{ active})$ drops **21%** (0.556→0.437), $E_{\rm chosen}$ essentially unchanged | the delay/authority-timing asymmetry is a real but *secondary* contributor (~1/5–1/4 of the effect); it does not explain the majority |
| Task-direction decomposition (§6.3) | insertion feedback is 90.5% axial ($x$), joint feedback is 95.5% lateral ($y$) | insertion's dominance is explained by genuine physical division of labor, not a cost-function pathology |

**Verdict:** insertion's high activity is a real, physically-grounded
property of this system (axial actuator, higher raw task efficiency,
partially reinforced by — but not caused by — the delay-authority timing
asymmetry), not a tuning artifact. $s_{u,L}=5$mm/s and $|u_L|\le2$mm/s are
kept at their original, physically-justified values. None of the three
insertion-suppressing interventions tested (forcing $\delta u_L=0$, the
$s_{u,L}$ rescale, zeroing early-stage $Q_p$) is carried into the frozen
live controller.

---

## 10. Hard constraints

**Frozen:** $|u_{q,i}|\le0.1$rad/s, $|u_L|\le2\times10^{-3}$m/s,
$|\Delta u_q|\le0.04$rad per 0.1s tick (≡ $0.4$rad/s² acceleration limit),
state box unchanged.

**Evidence:** these are the robot/advancer's own physical hardware limits
(not cost-function-derived), held fixed and identical across **every**
condition compared in this entire document. No ablation in this campaign
ever varied a hard constraint — only the cost weights around them — so
every table above is a legitimate like-for-like comparison of a fixed
feasible region under different objective functions.

---

## 11. Regression guard: γ=1 ↔ M0 equivalence

Kept permanently in the unit-test suite
(`stagewise_task_nullspace_self_test.py`), re-run as part of freezing this
controller (2026-09-20): setting the frozen $\gamma=0$ code path's own
$\gamma$ parameter to 1 reproduces M0's Hessian, linear term, and solved
$u_0$ to numerical precision ($u_0$ diff $\sim4\text{–}7\times10^{-16}$).
This matters structurally, not just as a sanity check: it proves the *same
code path* used live for $\gamma=0$ contains no unrelated implementation
difference from the already-validated M0 controller — the only degree of
freedom that changed is $\gamma$ itself.

The same principle governs the live wrapper
(`run_mpc_delay_aware_nullq_r700.py`): it constructs
`StagewiseTaskNullspaceDelayAwareMPC` directly, the exact class validated
above — no projector algebra was reimplemented in the hardware runner or
the worker process. The worker's own startup sequence additionally
re-verifies $P_N^2=P_N$, $P_N^T=P_N$, and $J^t_jS_zP_{N,j}\approx0$ over the
**full stored 265-entry schedule** before reporting ready (not just the one
representative index the unit test checks), and prints the resolved
controller configuration to console so the live run's own log proves which
controller actually executed.

---

## 12. Process isolation (execution architecture, not a cost-function term, but load-bearing)

**Frozen:** MPC solve happens in a separate `spawn`-context process
(`MPCWorkerHandle`/`worker_process.py`), warmed and signaled ready **before**
any RTDE/camera connection opens; at most one outstanding solve request; a
deadline miss applies $u=0$ (hold, don't advance the accumulator).

**Evidence:** `rectangle_stage_a/mpc_stationary_stress_test.py` (a
no-motion, 180s, 1800-tick stress test) established that a sustained run of
moderately heavy per-tick QP solves (5 consecutive ticks at 55–104ms, one
exceeding the full 100ms control period) directly precedes a **permanent**
freeze of the RTDE receive connection — confirmed by two independent
staleness checks aging in exact lockstep with wall-clock time from that
point on. A second failure mode was found and fixed during this project:
spawning the worker process is itself a real one-time CPU burst (fork +
fresh interpreter re-importing numpy/scipy/osqp), and landing that burst at
the same moment as an already-fragile camera/RTDE connect window reproduced
the same class of staleness failure — fixed by hard-ordering worker
spawn+warm-up strictly before any hardware connection. Since this fix: zero
deadline misses across the full offline replay campaign (536+ ticks) and
multiple live validation runs of the process-isolated M0/inverse
controllers.

---

## 13. Summary: every frozen parameter and its evidence source

| term/parameter | frozen value | evidence source (§) |
|---|---|---|
| Delay model | $d=2$ joints, insertion undelayed | §2 (5-fold LOO, $G=0.973$–$0.992$ re-derived; beats $d=0$ by 6–11× and a fitted fractional delay on 4/5 folds — `SUPPLEMENTARY_EVIDENCE.md` §S10) |
| Disturbance persistence | $\beta_d=1$ | §3 (sensitivity confirmed non-trivial — 16.3% command / 1.45mm prediction RMS shift vs $\beta_d=0.82$ — but not a superiority result; kept fixed to avoid confounding §6–§9 — §S11) |
| Horizon / step / terminal cost | $N=15$, $\Delta t=0.1$s, $V_f=0$ | §4 (convergence swept on predecessor LTV MPC: RMS flat from $N{=}10\to15$ on both rectangle and triangle, $N{=}1$ shown qualitatively unstable — §S12; held fixed across every ablation in this campaign; $V_f$ is a scope limit) |
| Task cost $Q_p$ | $1.0\,\mathrm{diag}(1/(0.5\mathrm{mm})^2)$, unchanged | §5 ($E_{\rm model}$ stayed 0.34–0.6mm everywhere — never the limiting factor) |
| State cost | exact task-nullspace $Q_N$, $\gamma=0$ | §6 (MT rejected; $\gamma$-sweep; $Q_N$-vs-off ablation) |
| Input tracking $R$ | $700\times R_0$ | §6.3/§7 (log sweep + refinement; R700≈R1000 on joint activity) |
| Input tracking channel scales | $s_{u,q}=0.05$rad/s, $s_{u,L}=5$mm/s, unchanged | §9 (rescale ablation: joints don't compensate, tracking worsens) |
| Input increment $R_d$ | $10^{-3}$, unchanged; scale explicitly pinned | §8 (auto-derive bug found + fixed; never implicated as binding) |
| Hard constraints | original hardware limits, untouched | §10 (held fixed everywhere, by design) |
| Execution model | execution-C accumulator, 50Hz streaming | §2 ($G$: 0.199→1.000; 4× error drop from streaming) |
| Process architecture | worker-process isolation, spawn-before-hardware | §12 (stress-test freeze mechanism + fix) |
| $\gamma=1\leftrightarrow$M0 regression guard | kept in CI permanently | §11 |

---

## 14. What is *not* yet established, and what the pending live run is for

This document justifies the **offline** case for this formulation. It does
not yet answer:

- Whether the axis-resolved $E_{\rm chosen}$ improvement (§6.3) survives
  contact with the real, live disturbance distribution — $E_{\rm chosen}$
  is an open-loop, within-horizon prediction proxy, replayed against a
  **fixed** historical log; it is not a closed-loop guarantee.
- Whether $|\bar e_y|$ (live lateral bias) actually falls, which is the
  single sharpest prediction this document makes and the primary thing the
  next live run is designed to test.
- Whether $E_{\rm model}$ stays low under the new formulation live (i.e.
  whether the controller stays inside the region the beam model was
  validated in) — if $E_{\rm chosen}$ improves live but $E_{\rm model}$
  degrades, that would indicate the controller is choosing trajectories
  outside the model's valid region, not that the reformulation "worked."

These are exactly the quantities `run_mpc_delay_aware_nullq_r700.py`'s
logging (predicted beam positions + `q_cmd_k`/`q_cmd_prev`/`z_meas`/`u_prev`/
`u0` every tick, plus `controller_metadata.json`) is designed to make
answerable after the run, without ambiguity about which controller produced
which log.
