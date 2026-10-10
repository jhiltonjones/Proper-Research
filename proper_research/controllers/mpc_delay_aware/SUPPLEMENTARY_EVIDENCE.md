# Supplementary evidence: raw tables for the delay-aware task-nullspace MPC (γ=0, R700)

This is the raw-data companion to `CONTROLLER_JUSTIFICATION.md`. Every table
below is the **unedited output** of a script re-run on 2026-09-20 from the
saved replay data in this investigation, so numbers here are reproducible
directly from the cited script rather than transcribed from memory.

**Common setup for every offline table (unless stated otherwise):**
- Plan: `plans/rectangle_10x15mm_skipglobal_2026-09-12/time_parameterized_configuration_path`
- Genuine per-sample LTV Jacobian schedule (265×3×7), cached
- Replay log: the 268-tick successful live run
  `close_loop_logs/myrun/mpc_delay_aware_accumC_20260918T165039Z`
  (`path_follow.jsonl` + `predicted_beam_positions.jsonl`)
- $N=15$, $\Delta t=0.1$s, $d=2$, $\beta_d=1$, $V_f=0$, hard constraints at
  their original hardware values — held fixed across every condition in
  every table, so each table isolates only the cost-function term(s) named
  in its heading.
- $E_{c,+j}=\mathrm{RMS}\|\hat p_{k+j|k}-p_{\rm des,k+j}\|$ in mm, using
  `predicted_beam_positions[j-1]` against `desired_position_m[idx+j]`.
- `P(u_active)` / `P(du_active)`: fraction of the 268 ticks where the
  applied $u_0$ / $\Delta u_0$ saturates any channel's hard bound.
- Units: joint velocities in mrad/s ("mrad" below), insertion in mm/s.

---

## S1. Unit / regression tests (raw output)

### S1.1 `stagewise_task_nullspace_self_test.py` — re-run 2026-09-20

```
[stagewise_task_nullspace self-test] 1/4 PASS: projector identities hold at every
horizon stage (symmetric, idempotent, P_R@P_N~=0, P_R+P_N~=I)
[stagewise_task_nullspace self-test] 2/4 PASS: pure row-space vectors are annihilated
by P_N; pure null-space vectors are annihilated by the scaled task Jacobian
[stagewise_task_nullspace self-test] 3/4 PASS: numerical rank = 2 at every stage
(computed, not assumed) -- nullspace is 5D
[stagewise_task_nullspace self-test] 4/4 PASS: gamma=1 reproduces M0 exactly
(H diff=1.86e-09, f diff=0.00e+00, u0 diff=6.94e-17)
[stagewise_task_nullspace self-test] ALL PASS
```

### S1.2 `target_consistent_integration_test.py` (MT's own regression suite — re-run 2026-09-20)

```
[target_consistent integration test] 1/4 PASS: H bit-identical between M0 and MT,
with and without disturbance
[target_consistent integration test] 2/4 PASS: zero disturbance -> MT bit-identical
to M0
[target_consistent integration test] 3/4 PASS: nonzero disturbance changes the linear
cost, exactly matching an independent per-stage target_consistent.py computation
[target_consistent integration test] 4/4 PASS: task residual ~0 at every horizon
stage (max=3.47e-18)
[target_consistent integration test] ALL PASS
```
(Kept as evidence that the tooling used to *reject* MT in §S2 was itself
correct — MT's rejection was a formulation result, not an implementation bug.)

### S1.3 Worker startup projector assertions, over the **full** 265-entry schedule (live-worker smoke test, `worker_process.py`)

```
[worker] nullspace projector startup checks PASS over full schedule (265, 3, 7):
idempotent_err=1.15e-15 symmetric_err=0.00e+00 task_annihilation_err=7.65e-19 ranks=[2]
[worker] state cost: exact task-nullspace only, gamma=0
[worker] R multiplier: 700 (input_tracking_weight=7.0)
[worker] Rd: baseline 0.001, increment scale explicitly pinned=
  (0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.002)
[worker] delay_samples=2 beta_d=1.0 horizon=15 terminal_cost=False (V_f=0)
[smoke] worker ready in 293ms
[smoke] nullspace_r700 solve: status=ok t_solve_ms=8.1
  command=[0.00593, 0.0002, -0.00854, -0.00635, 0.0053, 0.00126, 0.002]
[smoke] old (M0-undelayed) solve still works: status=ok
[smoke] new (M0 delay-aware) solve still works: status=ok
[smoke] ALL OK
```
Solve time 8.1ms is well inside the 70ms live deadline; all three controller
kinds (`old`/`new`/`nullspace_r700`) answer correctly from one worker.

---

## S2. MT (target-consistent) — rejection evidence

*(`mt_full_metrics.py`, `mt_diag123.py`, both re-run 2026-09-20 against
`mt_full_results.pkl`/`mt_replay_results.pkl`; conditions: `m0`, `m0_4qp`
= M0 with $Q_p\times4$, `m0_16qp` = M0 with $Q_p\times16$, `mt` =
MT-stagewise, `inv` = kp=1 delay-aware inverse.)*

### S2.1 Gradient/KKT decomposition at the first move (joints only, mrad, median over 268 ticks)

| controller | $\|g_{Q,0}\|$ | $\|g_{Qp,0}\|$ | $\|g_{R,0}\|$ | $\|g_{Rd,0}\|$ | $\|g_{constr,0}\|$ | $\cos(g_Q,g_{Qp})$ median | mean | p10 | p90 |
|---|---|---|---|---|---|---|---|---|---|
| m0 | 952849.1 | 952751.4 | 96.72 | 34.50 | 0.0 | −1.0000 | −0.9950 | −1.0000 | −1.0000 |
| mt | 228781.7 | 149489.8 | 494.88 | 114.82 | 2908.2 | −0.9998 | −0.7022 | −1.0000 | +0.3351 |

MT reduces the raw gradient magnitudes substantially but the $Q$/$Q_p$
antagonism (median cosine) barely moves off $-1$; the mean improves
(constraint activity pulls some ticks off the antagonistic extreme) but the
distribution's p90 of $+0.34$ shows this is inconsistent, not a clean fix —
this is the numeric basis for "none of the go/no-go criteria were cleanly
met."

### S2.2 $Q_p\times4/\times16$ vs MT — $E_{\rm chosen}$ / activity / effort

| cond | $E_{c,+3}$ | $E_{c,+5}$ | $E_{c,+10}$ | $P(\Delta u\text{ act})$ | $P(u\text{ act})$ | RMS($u_0{-}u_{\rm ref}$) mrad |
|---|---|---|---|---|---|---|
| m0 | 0.728 | 0.731 | 0.755 | 0.097 | 0.235 | 22.65 |
| m0_4qp | 0.470 | 0.419 | 0.430 | 0.366 | 0.634 | 49.78 |
| m0_16qp | 0.401 | 0.291 | 0.239 | 0.563 | 0.687 | 67.10 |
| mt | 0.411 | 0.275 | 0.161 | 0.552 | 0.709 | 64.89 |

MT's $E_{\rm chosen}$ is comparable to $Q_p\times16$'s but at similar (not
better) saturation cost — MT does not buy tracking improvement "for free"
relative to just brute-forcing $Q_p$ up, which is one of the pre-declared
negative criteria.

### S2.3 Stagewise constraint-activity profile, $j=0\ldots14$

```
m0 P(du_j active): 0.10 0.01 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
m0 P(u_j  active): 0.24 0.13 0.07 0.01 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
mt P(du_j active): 0.55 0.38 0.22 0.13 0.13 0.09 0.12 0.10 0.03 0.01 0.00 0.00 0.00 0.00 0.00
mt P(u_j  active): 0.71 0.78 0.70 0.63 0.60 0.51 0.45 0.38 0.33 0.30 0.24 0.20 0.17 0.09 0.06
```
MT's activity does not decay across the horizon the way a well-behaved
formulation's should — it stays elevated (>0.3) out to stage 8, "pinned"
rather than settling — the basis for "stagewise activity decaying, not
pinned" being unmet.

### S2.4 MT-vs-inverse(kp=1) task-correction alignment (condensed config space)

```
cos(P_R*du_MT, P_R*du_INV): median=0.6770 mean=0.1058 p10=-0.9916 p90=0.9984  n=268
```
Wide, bimodal-looking spread (large positive p90 but strongly negative p10) —
not a clean directional match.

### S2.5 Diagnostic 1 — is MT's stagewise target dynamically consistent?

```
rho^t_u  (implied target velocity / limit): median=0.776 p95=0.863 max=1.217 P(>1)=0.019
rho^t_du (implied target accel   / limit):   median=0.047 p95=0.499 max=0.937 P(>1)=0.000
RMS||u^t_j - u_ref,j|| over the horizon: median-per-tick=8.41mrad p95=41.23mrad max=56.49mrad
```
The target-implied velocity occasionally exceeds the hard velocity limit
itself ($P(\rho^t_u>1)=1.9\%$ of ticks) — the per-stage target is not always
even kinematically realizable, let alone consistent with $u_{\rm ref}$.

### S2.6 Diagnostic 2 — state-space (per-stage $z$) $Q$ vs $Q_p$ gradient cosine, all 15 stages

| stage $j$ | median $\cos(g_Q^z,g_{Qp}^z)$ | mean |
|---|---|---|
| 1 | +0.9915 | +0.8290 |
| 2 | +0.9815 | +0.7295 |
| 3 | +0.9643 | +0.3586 |
| 4 | +0.0718 | +0.1561 |
| 5 | +0.0568 | +0.0590 |
| 6 | +0.0572 | −0.0234 |
| 7 | −0.0647 | −0.0886 |
| 8 | −0.6659 | −0.2157 |
| 9 | −0.9880 | −0.3075 |
| 10 | −0.9988 | −0.4069 |
| 11 | −0.9993 | −0.5992 |
| 12 | −0.9999 | −0.6771 |
| 13 | −0.9999 | −0.7328 |
| 14 | −1.0000 | −0.7901 |
| 15 | −1.0000 | −0.8532 |

Locally correct (near-agreement) at the first 2–3 stages, then the gradients
flip to strongly antagonistic by mid-horizon — confirming "local correctness
but horizon-wide inconsistency," which motivated testing MT-constant
(§S2 continuation, not separately tabled: MT-constant behaved almost
identically to MT-stagewise in §S3.1 rows 3–4, disproving the dynamic-
inconsistency hypothesis as the dominant driver).

### S2.7 Diagnostic 3 — MT-vs-INV alignment in TASK-DISPLACEMENT space (not scaled config space)

```
cos(dp_MPC,+3, dp_INV,+3) [raw commanded task displacement]:    median=0.5271 mean=0.4205 p10=-0.3388 n=268
cos(dp_MPC,+3, dp_INV,+3) [correction only, u-u_ref]:           median=0.3346 mean=0.1243 p10=-0.9750 n=268
```
Corrected, physically-meaningful alignment metric is weaker than the
condensed-space one (§S2.4) once measured directly in task-displacement
space — reinforcing the decision to stop developing MT and pursue the
structurally different $Q_\gamma$ formulation instead.

---

## S3. Task/nullspace $Q_\gamma$ — 6-condition sweep

*(`mt_six_condition_metrics.py`, re-run 2026-09-20; conditions:
M0, M0+4$Q_p$, MT-stagewise, MT-constant, $Q_{\gamma=0.25}$, $Q_{\gamma=0.10}$.)*

### S3.1 $E_{\rm chosen}$ / activity / effort

| cond | $E_{c,+3}$ | $E_{c,+5}$ | $E_{c,+10}$ | $P(\Delta u\text{ act})$ | $P(u\text{ act})$ | RMS($u_0{-}u_{\rm ref}$) mrad | RMS($\Delta u_0$) mrad |
|---|---|---|---|---|---|---|---|
| 1.M0 | 0.728 | 0.731 | 0.755 | 0.097 | 0.235 | 22.65 | 32.38 |
| 2.M0+4Qp | 0.470 | 0.419 | 0.430 | 0.366 | 0.634 | 49.78 | 52.14 |
| 3.MT-stagewise | 0.411 | 0.275 | 0.161 | 0.552 | 0.709 | 64.89 | 65.25 |
| 4.MT-constant | 0.411 | 0.277 | 0.197 | 0.556 | 0.713 | 64.91 | 65.27 |
| 5.Qγ=0.25 | 0.475 | 0.420 | 0.430 | 0.351 | 0.631 | 47.29 | 49.76 |
| 6.Qγ=0.10 | 0.420 | 0.324 | 0.287 | 0.519 | 0.687 | 58.13 | 58.98 |

MT-stagewise vs MT-constant (rows 3–4): nearly identical on every column —
the direct evidence that per-stage target dynamics were not the dominant
saturation driver (§S2.6's diagnosis notwithstanding, its practical effect
is small). $Q_{\gamma=0.10}$ (row 6) is the best Pareto point of this batch:
similar $E_{c,+10}$ to $Q_p\times4$ (row 2) at comparable-to-better activity,
without MT's structural issues.

### S3.2 Stagewise $P(u_j\text{ active})$, all 6 conditions

```
1.M0            : 0.24 0.13 0.07 0.01 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
2.M0+4Qp        : 0.63 0.49 0.36 0.24 0.18 0.10 0.06 0.01 0.00 0.00 0.00 0.00 0.00 0.00 0.00
3.MT-stagewise  : 0.71 0.78 0.70 0.63 0.60 0.51 0.45 0.38 0.33 0.30 0.24 0.20 0.17 0.09 0.06
4.MT-constant   : 0.71 0.78 0.69 0.62 0.58 0.49 0.38 0.30 0.25 0.22 0.18 0.14 0.10 0.08 0.05
5.Qgamma=0.25   : 0.63 0.48 0.37 0.24 0.18 0.10 0.06 0.01 0.00 0.00 0.00 0.00 0.00 0.00 0.00
6.Qgamma=0.10   : 0.69 0.70 0.51 0.41 0.34 0.28 0.22 0.16 0.12 0.06 0.03 0.01 0.00 0.00 0.00
```
$Q_{\gamma=0.10}$ (row 6) decays across the horizon the way MT does not
(§S2.3) — this decay pattern is what made it "the standout" of the sweep.

---

## S4. $R$/$R_d$ isolation at fixed $\gamma=0.10$

*(`mt_rrd_metrics.py`, re-run 2026-09-20, post-bugfix — `input_increment_scale`
explicitly pinned; see `CONTROLLER_JUSTIFICATION.md` §8 for the bug this
fix corrects.)*

### S4.1 $E_{\rm chosen}$ / activity / effort / unconstrained-optimum diagnostic

| cond | $E_{c,+3}$ | $E_{c,+5}$ | $E_{c,+10}$ | $P(\Delta u\text{ act})$ | $P(u\text{ act})$ | RMS($u_0{-}u_{\rm ref}$) | RMS($\Delta u_0$) | $\rho_{\rm desire}$ med | p95 | max |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.420 | 0.324 | 0.287 | 0.519 | 0.687 | 58.13 | 58.98 | 1.605 | 2.673 | 3.535 |
| R3 | 0.422 | 0.324 | 0.287 | 0.489 | 0.683 | 56.56 | 57.62 | 1.578 | 2.619 | 3.436 |
| R10 | 0.431 | 0.327 | 0.288 | 0.414 | 0.675 | 51.87 | 53.57 | 1.473 | 2.463 | 3.303 |
| Rd3 | 0.421 | 0.323 | 0.287 | 0.504 | 0.687 | 57.50 | 58.37 | 1.566 | 2.612 | 3.418 |
| Rd10 | 0.424 | 0.323 | 0.287 | 0.474 | 0.679 | 55.42 | 56.42 | 1.488 | 2.435 | 3.199 |

$R\times10$ and $R_d\times10$ each move $P(u\text{ active})$ by only
0.01–0.02 and $E_{c,+10}$ essentially not at all — the mild-effects result
that (correctly, per the bugfix) motivated abandoning $\gamma=0.10$ as the
tuning point for $R$ and testing $\gamma=0$ instead (§S5).

### S4.2 Stagewise activity, same 5 conditions

```
P(u_j active):
baseline  : 0.69 0.70 0.51 0.41 0.34 0.28 0.22 0.16 0.12 0.06 0.03 0.01 0.00 0.00 0.00
R3        : 0.68 0.70 0.51 0.41 0.34 0.27 0.22 0.16 0.12 0.06 0.03 0.01 0.00 0.00 0.00
R10       : 0.68 0.68 0.50 0.41 0.33 0.27 0.22 0.16 0.12 0.05 0.03 0.01 0.00 0.00 0.00
Rd3       : 0.69 0.70 0.51 0.41 0.34 0.28 0.22 0.16 0.12 0.06 0.03 0.01 0.00 0.00 0.00
Rd10      : 0.68 0.69 0.50 0.41 0.33 0.27 0.22 0.16 0.12 0.05 0.03 0.01 0.00 0.00 0.00

P(du_j active):
baseline  : 0.52 0.17 0.06 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
R3        : 0.49 0.12 0.04 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
R10       : 0.41 0.06 0.01 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Rd3       : 0.50 0.14 0.05 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Rd10      : 0.47 0.08 0.03 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
```

---

## S5. Exact task-nullspace, $\gamma=0$ — $R$ scale-finding and refinement

*(`gamma0_R_metrics.py` and `gamma0_R_refine_metrics.py`, re-run 2026-09-20.)*

### S5.1 Logarithmic scan, $R\times\{1,10,100,1000\}$, $\gamma=0$ fixed

| cond | $E_{c,+3}$ | $E_{c,+5}$ | $E_{c,+10}$ | $P(\Delta u\text{ act})$ | $P(u\text{ act})$ | RMS($u_0{-}u_{\rm ref}$) | $\rho_d$ med | p95 | max |
|---|---|---|---|---|---|---|---|---|---|
| R1 | 0.413 | 0.285 | 0.167 | 0.549 | 0.690 | 58.12 | 2.333 | 6.280 | 8.647 |
| R10 | 0.419 | 0.287 | 0.173 | 0.500 | 0.687 | 55.14 | 2.118 | 3.639 | 4.515 |
| R100 | 0.464 | 0.319 | 0.207 | 0.347 | 0.660 | 40.59 | 1.420 | 2.389 | 2.996 |
| R1000 | 0.576 | 0.427 | 0.298 | 0.101 | 0.511 | 18.21 | 0.661 | 1.133 | 1.645 |

$R1000$ reaches $\rho_{\rm desire}\approx0.66$ (comfortably under the hard
limit even in the unconstrained-optimum sense) with $E_{c,+10}=0.298$mm —
this is the row that motivated refining between R100 and R1000 rather than
jumping straight to R1000 as final.

### S5.2 Refinement, $R\times\{1,100,300,500,700,1000\}$

| cond | $E_{c,+3}$ | $E_{c,+5}$ | $E_{c,+10}$ | $P(\Delta u\text{ act})$ | $P(u\text{ act})$ | RMS($u_0{-}u_{\rm ref}$) | RMS($\Delta u_0$) | $\rho_d$ med | p95 | max |
|---|---|---|---|---|---|---|---|---|---|---|
| R1 | 0.413 | 0.285 | 0.167 | 0.549 | 0.690 | 58.12 | 58.70 | 2.333 | 6.280 | 8.647 |
| R100 | 0.464 | 0.319 | 0.207 | 0.347 | 0.660 | 40.59 | 44.23 | 1.420 | 2.389 | 2.996 |
| R300 | 0.513 | 0.357 | 0.243 | 0.209 | 0.619 | 28.94 | 35.04 | 1.023 | 1.705 | 2.131 |
| R500 | 0.539 | 0.383 | 0.264 | 0.157 | 0.590 | 24.03 | 31.40 | 0.834 | 1.420 | 1.906 |
| **R700** | **0.557** | **0.404** | **0.279** | **0.127** | **0.556** | **21.00** | **29.30** | **0.750** | **1.255** | **1.772** |
| R1000 | 0.576 | 0.427 | 0.298 | 0.101 | 0.511 | 18.21 | 27.49 | 0.661 | 1.133 | 1.645 |

R700 vs R1000: $E_{c,+10}$ 0.279 vs 0.298mm, $\rho_{\rm desire}$ median 0.750
vs 0.661 — R700 gives up very little relative to R1000 while being a
smaller, less aggressive multiplier; this pairing is the direct basis for
selecting R700 over R1000 as the frozen value.

### S5.3 Stagewise activity, refinement set

```
P(u_j active):
R1     : 0.69 0.77 0.72 0.61 0.56 0.46 0.43 0.37 0.33 0.30 0.24 0.20 0.17 0.09 0.06
R100   : 0.66 0.74 0.57 0.49 0.46 0.41 0.37 0.32 0.30 0.23 0.20 0.17 0.15 0.07 0.06
R300   : 0.62 0.65 0.50 0.45 0.41 0.37 0.32 0.29 0.25 0.20 0.17 0.15 0.11 0.06 0.06
R500   : 0.59 0.52 0.47 0.44 0.40 0.36 0.29 0.27 0.21 0.18 0.15 0.13 0.11 0.06 0.06
R700   : 0.56 0.48 0.45 0.40 0.38 0.34 0.28 0.24 0.20 0.17 0.13 0.11 0.07 0.06 0.05
R1000  : 0.51 0.46 0.42 0.39 0.35 0.31 0.27 0.22 0.16 0.14 0.11 0.08 0.07 0.05 0.04

P(du_j active):
R1     : 0.55 0.30 0.11 0.04 0.02 0.04 0.01 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
R100   : 0.35 0.01 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
R300   : 0.21 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
R500   : 0.16 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
R700   : 0.13 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
R1000  : 0.10 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
```
$P(\Delta u\text{ active})$ collapses to ~0 beyond stage 1 for every
$R\ge100$ — rate saturation is confined to the very first move even before
refinement, another reason $R_d$ was never separately implicated (see
`CONTROLLER_JUSTIFICATION.md` §8).

### S5.4 Per-actuator activity + headroom (R100, R500, R1000) — the discovery table

```
--- R100 ---
  q1: P(active)=0.030  rho(u_ref)=0.698  headroom=30.21mrad  |delta_u|=30.21mrad
  q2..q6: P(active)=0.000 (every tick, every one of these 5 channels)
  L:  P(active)=0.634  rho(u_ref)=0.652  headroom=0.70mm/s   |delta_u|=0.71mm/s

--- R500 ---
  q1: P(active)=0.004  rho(u_ref)=0.528  headroom=47.19mrad  |delta_u|=47.19mrad
  q2..q6: P(active)=0.000
  L:  P(active)=0.586  rho(u_ref)=0.665  headroom=0.67mm/s   |delta_u|=0.67mm/s

--- R1000 ---
  q1..q6: P(active)=0.000 (every one of these 6 channels)
  L:  P(active)=0.511  rho(u_ref)=0.730  headroom=0.54mm/s   |delta_u|=0.54mm/s
```
This is the table that reclassified every "constraint activity" number
reported earlier in the campaign: at R500/R1000, **zero of the six joint
channels ever saturate**; the aggregate $P(u\text{ active})$ numbers in every
table above (S3, S4, S5.1, S5.2) are driven almost entirely by the insertion
channel $L$.

---

## S6. Insertion authority audit

### S6.1 R700/γ=0 MPC vs kp=1 inverse — insertion-specific diagnostics (`insertion_audit.py`)

```
--- R700 MPC (gamma=0) ---
  rho_ref,L:       median=0.777 p95=0.777 max=0.780
  |delta_u_L|:     RMS=0.9337mm/s  p95=1.9865mm/s  max=2.0686mm/s
  headroom h_L:    median=0.4462mm/s  p95=2.0000mm/s
  rho_fb/headroom: median=1.000  p95=1.037  max=2.174
  P(u_L active):   0.556
  at saturated ticks: same-direction (feedback pushes further) = 0.919,
                       opposite-direction (feedback opposes planner) = 0.081

--- INV kp=1 (delay-aware) ---
  rho_ref,L:       median=0.777 p95=0.777 max=0.780   (identical planner reference, as expected)
  |delta_u_L|:     RMS=0.0000mm/s  p95=0.0000mm/s  max=0.0000mm/s
  headroom h_L:    median=0.4462mm/s  p95=2.0000mm/s
  rho_fb/headroom: median=0.000  p95=0.000  max=0.000
  P(u_L active):   0.000
```
$\rho_{\rm ref,L}$ (the planner's own reference insertion velocity relative
to the hard limit) is identical for both controllers, as it must be — the
divergence is entirely in $\delta u_L$, the feedback correction. 91.9% of
R700's saturated ticks are same-direction (the feedback is *reinforcing* the
planner's own insertion motion, not fighting it) — evidence against
"insertion feedback is erratic/fighting the plan."

### S6.2 Task-contribution decomposition, R700 (FREE-BOTH/unconstrained)

$$\eta_L=\frac{\|J_L\Delta L\|}{\|J_q\Delta q\|+\|J_L\Delta L\|+\epsilon}$$

```
eta_L (fraction of task-correction norm from insertion): median=0.278 p95=0.949 mean=0.396
```
Insertion is not always dominant (median 0.278) but at the tail it carries
almost the entire task correction (p95 = 0.949) — consistent with insertion
being reserved for exactly the moments joints have little authority or high
task cost to act, which §6.5/S6.4 explains structurally.

### S6.3 $s_{u,L}$ rescale ablation (5mm/s → 2mm/s, joints untouched)

```
--- sul5_default ---
  Ec+3=0.557 Ec+5=0.404 Ec+10=0.279
  P(u_L active)=0.556  P(u_q active, any joint)=0.000  P(du_q active)=0.037
  RMS(u0-uref) joints only = 20.98mrad
  RMS(u0-uref) insertion   = 0.9337mm/s
--- sul2 ---
  Ec+3=0.583 Ec+5=0.429 Ec+10=0.293
  P(u_L active)=0.384  P(u_q active, any joint)=0.000  P(du_q active)=0.037
  RMS(u0-uref) joints only = 20.99mrad
  RMS(u0-uref) insertion   = 0.5550mm/s
```
Making insertion 2.5× more expensive in $R$-normalized terms reduces its
usage (P(active) 0.556→0.384, RMS 0.934→0.555mm/s) — but joint usage is
**completely unchanged** (RMS 20.98→20.99mrad, to the second decimal) and
$E_{\rm chosen}$ gets *worse* across the board. This is the direct evidence
ruling out "insertion is simply mis-costed relative to its headroom": if it
were, suppressing it should have shifted the burden onto joints with little
or no tracking cost; instead the burden was simply dropped.

### S6.4 Per-stage first-move authority ($\|Q_p^{1/2}\partial\hat p_j/\partial u_{i,0}\|$)

| stage | $\|Q_p^{1/2}\partial\hat p_j/\partial u_{q,\rm avg,0}\|$ | $\|Q_p^{1/2}\partial\hat p_j/\partial u_{L,0}\|$ |
|---|---|---|
| 1 | 0.0000 | 201.0108 |
| 2 | 0.0000 | 201.1260 |
| 3 | 32.8514 | 201.2133 |
| 4 | 32.9032 | 201.2743 |
| 5 | 32.9502 | 201.3085 |
| 6 | 33.0245 | 201.3190 |
| 7 | 33.0843 | 201.2743 |
| 8 | 33.1399 | 201.2026 |
| 9 | 33.2160 | 201.0895 |
| 10 | 33.2787 | 200.9524 |
| 11 | 33.3661 | 200.7616 |
| 12 | 33.4496 | 200.5681 |
| 13 | 33.5605 | 200.3072 |
| 14 | 33.6492 | 200.0543 |
| 15 | 34.1909 | 200.0280 |

Joint authority is **exactly** 0.0000 at stages 1–2 (a structural property of
$d=2$ — a joint decision made now has zero effect on physical joint position
until stage 3) and insertion authority is essentially constant (~200–201) at
**every** stage, roughly 6× the joint authority even once joints become
active from stage 3 on.

### S6.5 Cost-normalized authority per actuator, $A_i=\|Q_p^{1/2}G_{0,i}\|/\sqrt{R_{ii}}$, summed over the horizon

| actuator | raw authority (Σ over horizon) | $/\sqrt{R_{ii}}$ |
|---|---|---|
| q1 | 47.0899 | 0.889916 |
| q2 | 71.9922 | 1.360524 |
| q3 | 67.9894 | 1.284879 |
| q4 | 28.4587 | 0.537819 |
| q5 | 38.6208 | 0.729865 |
| q6 | 8.7843 | 0.166007 |
| **L** | **778.0816** | **38.904081** |

Even after normalizing by each channel's own $R$-cost, insertion's
cost-normalized authority is **~29–235× larger than any single joint's**.
This is the single clearest number in the whole investigation for "insertion
dominance is not a weighting artifact" — it holds even after accounting for
$R$.

### S6.6 Per-actuator activity across independently-run earlier conditions (M0, MT, $Q_{\gamma=0.10}$) — confirms the S5.4 finding held all day, not just at R700

```
--- m0 ---
  q1: P(active)=0.011   q2-q6: P(active)=0.000 (all five)   L: P(active)=0.224
--- mt ---
  q1: P(active)=0.045   q2-q6: P(active)=0.000 (all five)   L: P(active)=0.672
--- qg10 ---
  q1: P(active)=0.045   q2-q6: P(active)=0.000 (all five)   L: P(active)=0.646
```
$q_2$–$q_6$ are exactly 0.000 in **every** condition tested across the
entire campaign (M0, MT, $Q_{\gamma=0.10}$, and the full $R$-sweep in
S5.4) — only $q_1$ ever shows nonzero (small) activity, and insertion is the
only channel that saturates substantially, in every formulation tried.

### S6.7 Actuator-substitution ablations (`insertion_mechanism_metrics.py`)

| cond | success | $E_{c,+3}$ | $E_{c,+5}$ | $E_{c,+10}$ | $P(\Delta u_q\text{ act})$ | $P(u_L\text{ act})$ | RMS joints (mrad) | RMS insertion (mm/s) |
|---|---|---|---|---|---|---|---|---|
| R700_baseline | 1.000 | 0.557 | 0.404 | 0.279 | 0.037 | 0.556 | 20.98 | 0.9337 |
| jointonly ($\delta u_L{=}0$ forced) | 1.000 | 0.676 | 0.592 | **0.571** | 0.041 | 0.000 | 21.08 | 0.0000 |
| authaware ($Q_{p,1}{=}Q_{p,2}{=}0$) | 1.000 | 0.563 | 0.405 | 0.279 | 0.037 | 0.437 | 20.98 | 0.6965 |

Forcing zero insertion feedback more than doubles $E_{c,+10}$ (0.279→0.571)
while joint effort is **statistically unchanged** (20.98→21.08mrad,
0.037→0.041) — the optimizer does not substitute joint effort for lost
insertion authority; it simply accepts worse tracking. Removing the
early-stage-only task cost (where joints structurally lack authority) only
reduces insertion usage by ~21% (0.556→0.437) with no change to tracking —
a real but secondary contributor, confirming §S6.4/S6.5's "raw efficiency,
not just timing" conclusion.

---

## S7. Axis-resolved $E_{\rm chosen}$ and task-direction decomposition, M0 vs R700/γ=0

*(`axis_decomp_metrics.py`, re-run 2026-09-20; $C$ = beam-plane projection,
row0 = axial/$x$, row1 = normal/$y$.)*

### S7.1 Axis-resolved $E_{\rm chosen}$ (mm)

| cond | $j$ | $E_{c,x}$ | $E_{c,y}$ | $E_{c,\rm tot}$ |
|---|---|---|---|---|
| M0 | 3 | 0.419 | 0.593 | 0.728 |
| M0 | 5 | 0.425 | 0.592 | 0.731 |
| M0 | 10 | 0.444 | 0.607 | 0.755 |
| R700 | 3 | 0.264 | 0.486 | 0.557 |
| R700 | 5 | 0.201 | 0.335 | 0.404 |
| R700 | 10 | **0.094** | **0.238** | 0.279 |

M0's $y$-error is flat (0.593→0.607mm) across the whole horizon — it never
converges; R700/γ=0 drops $y$ 61% and $x$ 79% by stage +10.

### S7.2 Task-space direction decomposition (stage-0 Jacobian, R700)

| feedback source | RMS$_x$ | RMS$_y$ | mean$\|x\|$ | mean$\|y\|$ | frac$\|x\|$ | frac$\|y\|$ |
|---|---|---|---|---|---|---|
| $\Delta p_q$ (joint feedback) | 0.0529mm | 0.1558mm | 0.0300mm | 0.1233mm | 0.212 | **0.955** |
| $\Delta p_L$ (insertion feedback) | 0.0903mm | 0.0257mm | 0.0685mm | 0.0179mm | **0.905** | 0.278 |

($n=268$ nonzero ticks for both.) Joint feedback is 95.5% lateral, insertion
feedback is 90.5% axial — the physical division-of-labor pattern predicted
before this diagnostic was run.

---

## S8. $Q_N$-necessity ablation (full $u_{\rm ref}$, R700/γ=0, $Q_N$ on vs off)

*(`qn_ablation_metrics.py`, re-run 2026-09-20.)*

### S8.1 Task tracking and actuator effort

| cond | $j$ | $E_{c,x}$ | $E_{c,y}$ | $E_{c,\rm tot}$ |
|---|---|---|---|---|
| A ($Q_N$ on) | 3 | 0.264 | 0.486 | 0.557 |
| A | 5 | 0.201 | 0.335 | 0.404 |
| A | 10 | 0.094 | 0.238 | 0.279 |
| B ($Q_N=0$) | 3 | 0.264 | 0.482 | 0.552 |
| B | 5 | 0.201 | 0.325 | 0.387 |
| B | 10 | 0.094 | 0.227 | 0.248 |

| | A ($Q_N$ on) | B ($Q_N=0$) |
|---|---|---|
| $P(\Delta u_q\text{ active})$ | 0.037 | 0.037 |
| $P(u_L\text{ active})$ | 0.556 | 0.552 |
| RMS joints (mrad) | 20.98 | 21.69 |
| RMS insertion (mm/s) | 0.9337 | 0.9339 |

Task tracking and effort are essentially indistinguishable — B is even
marginally *better* on aggregate $E_{\rm chosen}$.

### S8.2 Accumulated nullspace configuration deviation $e_N(k)=P_{N,j}S_z^{-1}(z_{j|k}-z_{\rm ref,k+j})$ (normalized config units)

| stage | RMS$\|e_N\|$ A | max$\|e_N\|$ A | mean$\|e_N\|$ A | RMS$\|e_N\|$ B | max$\|e_N\|$ B | mean$\|e_N\|$ B |
|---|---|---|---|---|---|---|
| +3 | 0.1737 | 0.4328 | 0.1395 | 0.2077 | 0.5120 | 0.1729 |
| +10 | 0.2757 | 0.7670 | 0.2254 | **0.5244** | 1.1644 | 0.4595 |
| +15 | 0.2896 | 0.7624 | 0.2319 | **0.6945** | **1.8881** | 0.6001 |

Without $Q_N$, predicted nullspace drift is ~1.2× at stage+3, ~1.9× at
stage+10, and ~2.4× (RMS) / ~2.5× (max) at the terminal stage — despite
task tracking (S8.1) being unaffected. This is the table that directly
overturns the "full $u_{\rm ref}$ already makes $Q_N$ redundant" hypothesis.

---

## S10. Delay model: $d=2$, 5-fold leave-one-out identification

*(Raw per-fold data from `delay_model_results.pkl`/`tau_loo.pkl`, produced in
an earlier phase of this project — the original fitting script is no longer
present in this job's scratch directory, so the table below is read directly
from the saved fold results rather than re-run; the $G_{d=2}$/$G_{\rm frac}$
columns are my own transparent recomputation from those saved numbers, not a
re-transcription of the docstring's summary figure, so a reader can check the
formula: $G=1-(\mathrm{RMSE}_{\rm held\text{-}out}/\mathrm{RMSE}_{m0,\rm held\text{-}out})^2$.)*

Three joint-tracking models were fit per fold and evaluated on the held-out
fold: **m0** = no-delay model ($q_{\rm phys,k}=q_{{\rm cmd},k}$), **m2** =
the frozen $d=2$ model ($q_{\rm phys,k}=q_{{\rm cmd},k-2}$), **mf** = a
fractional-delay model ($\tau\approx2.16$ samples, fit per fold — see
`tau_loo` below).

### S10.1 Per-fold train/held-out RMSE (mrad) and $G$

| fold | m0 train | m0 held-out | m2 train | m2 held-out | mf train | mf held-out | $G_{d=2}$ | $G_{\rm frac}$ |
|---|---|---|---|---|---|---|---|---|
| 0 | 3.3246 | 5.6382 | 0.7824 | 0.9335 | 0.9016 | 0.9636 | 0.9726 | 0.9708 |
| 1 | 3.4682 | 8.3781 | 0.4127 | 0.8286 | 0.4156 | 1.0386 | 0.9902 | 0.9846 |
| 2 | 3.7593 | 9.4145 | 0.5204 | 0.8662 | 0.3620 | 0.5671 | 0.9915 | 0.9964 |
| 3 | 3.5637 | 9.1949 | 0.5177 | 1.0867 | 0.4798 | 1.1660 | 0.9860 | 0.9839 |
| 4 | 3.4136 | 7.4345 | 0.6123 | 0.9362 | 0.6910 | 1.0540 | 0.9841 | 0.9799 |

$G_{d=2}$ ranges **0.973–0.992** across the 5 held-out folds (consistent
with — though, on this exact re-derivation, not bit-identical to — the
"$G_{\rm delay}=0.945$–$0.986$" figure quoted in `prediction.py`'s and
`inverse_jacobian_2dof_delay_aware.py`'s docstrings; both agree the $d=2$
model explains essentially all the delay-induced tracking residual on every
held-out fold). The $d=2$ model's held-out RMSE (0.83–1.09mrad) is roughly
**6–11× smaller** than the no-delay model's held-out RMSE (5.6–9.4mrad) on
every fold — a large, consistent effect, not a marginal one.

Per-fold fitted fractional delay $\tau$ (`tau_loo.pkl`): folds 0–4 give
$\tau=$ 0.218, 0.217, 0.214, 0.216, 0.217 samples — tightly clustered, but
note this parameterization is **not** the same "2.16 samples" the docstring
refers to for the fractional model `mf` (that model's own per-fold $\tau$
isn't separately saved in this file); read this row only as "the fractional
delay parameter was small and consistent across folds," not as a
reproduction of the specific 2.16-sample figure.

### S10.2 Fractional model generalization check

$G_{\rm frac}$ is close to $G_{d=2}$ on 4/5 folds but the fractional model's
raw held-out RMSE is **worse** than the integer $d=2$ model's on **4 of 5
folds** (fold 2 is the exception, where `mf` held-out RMSE 0.567mrad beats
`m2`'s 0.866mrad). This matches the docstring's qualitative claim that the
fractional model "did not generalize" as reliably as the integer $d=2$
model, though the exact fold count differs slightly from the "negative on 3
of 5" wording — directionally the same conclusion (integer $d=2$ is the more
robust, reproducible choice), not a numerically identical re-derivation.

**Conclusion:** $d=2$ is the best-supported integer delay across every
held-out fold, by a wide margin over $d=0$, and is at least as robust as the
fractional alternative while being simpler (one integer parameter vs a
per-fold-fitted $\tau$). This is real cross-validated evidence, not an
assumption — but it was established prior to, and is not re-derived within,
this specific state-cost/R-tuning campaign (§S2–S8).

---

## S11. Disturbance persistence $\beta_d$: sensitivity of $\beta_d{=}1.0$ vs $\beta_d{=}0.82$

*(`mpc_beta_sensitivity.py`, re-run 2026-09-20 against a 262-tick live MPC-LTV
run, `close_loop_logs/rectangle_mpc_ltv_accumC_2026-09-17/mpc_ltv_accumC_rep2_genuineLTV_20260917T181546Z`,
delay-aware controller at $d=2$, both constrained and FREE-BOTH variants.)*

### S11.1 FREE-BOTH unconstrained-optimum aggressiveness ($\rho_{\rm desire}$)

```
beta=1.0:  median=4.641  p95=7.705  max=11.053  P(rho>1)=1.000
beta=0.82: median=4.593  p95=7.129  max=10.105  P(rho>1)=1.000
```
(This particular replay log is itself an aggressive/high-disturbance MPC-LTV
run where $\rho_{\rm desire}>1$ essentially always — the comparison here is
relative between the two $\beta_d$ values on identical inputs, not an
absolute aggressiveness statement about the frozen controller.)

### S11.2 Constrained stagewise activity

```
beta=1.0  P(du_j active): 1.00 0.97 0.97 0.76 0.73 0.39 0.18 0.04 0.01 0.00 0.00 0.00 0.00 0.00 0.00
beta=1.0  P(u_j  active): 0.53 0.55 0.37 0.34 0.32 0.24 0.20 0.11 0.08 0.05 0.03 0.02 0.02 0.01 0.00
beta=0.82 P(du_j active): 1.00 0.97 0.94 0.76 0.73 0.40 0.16 0.02 0.00 0.00 0.00 0.00 0.00 0.00 0.00
beta=0.82 P(u_j  active): 0.52 0.48 0.34 0.30 0.27 0.21 0.11 0.07 0.05 0.02 0.02 0.01 0.00 0.00 0.00
```

### S11.3 Direct controller and prediction sensitivity to $\beta_d$

```
RMS||u0-u_ref||: beta=1.0: 79.735mrad   beta=0.82: 77.902mrad

Delta_u_beta = RMS||u0^0.82 - u0^1.0||         = 14.3049mrad
RMS||u0^1.0||                                   = 87.5901mrad
S_beta = Delta_u_beta / RMS||u0^1.0||           = 0.1633  (16.33%)

per-actuator RMS(u0^0.82 - u0^1.0):
  q1: 6.8959mrad   q2: 5.7981mrad   q3: 6.0298mrad
  q4: 6.0471mrad   q5: 6.6780mrad   q6: 2.4343mrad
  L:  0.1103mm

Delta_p_beta = RMS||p_stack^0.82 - p_stack^1.0|| (full N=15 stack) = 1.4528mm
Delta_p_beta (stage j=1 only)                                       = 0.5401mm
```

**Interpretation:** switching $\beta_d$ from 1.0 to 0.82 changes the
commanded solution by a non-trivial 16.3% (RMS) and the predicted beam-tip
trajectory by up to 1.45mm RMS over the horizon — so this is **not** an
inconsequential parameter, and freezing it deserves the same scrutiny as any
other term. This table is a **sensitivity** result, not a **superiority**
result: it shows $\beta_d$ matters, not that 1.0 is better than 0.82. Per
`delay_aware_mpc.py`'s own docstring, $\beta_d=0.82$ was separately
identified as a better fit to the disturbance's actual decay behavior but
deliberately **not yet adopted** — $\beta_d=1$ (constant persistence, the
conservative/worst-case assumption) was kept as the frozen first-ablation
value specifically so that every state-cost/$R$ comparison in §S2–S8 was not
confounded by a second, simultaneously-changing disturbance-model
assumption. Adopting $\beta_d=0.82$ remains a documented, evidenced,
*open* refinement for a future iteration, not a rejected alternative.

---

## S12. Prediction horizon $N$: convergence sweep

*(Two independent sweeps against the earlier, non-delay-augmented LTV MPC
predecessor — `mpc_q_and_horizon_sweep.py`/`.out` and `horizon_sweep_dk.py`/
`.out`. The delay-aware augmentation changes the state representation, not
the horizon-length tradeoff itself — the beam+robot dynamics and
linearization-error-vs-lookahead tradeoff these sweeps characterize are the
same ones $N$ has to be chosen for in the delay-aware controller — but note
these specific numbers predate the delay-aware formulation and were not
re-run with it.)*

### S12.1 Horizon sweep, single reference trajectory (Q=0.001 fixed)

```
N= 1   rms=0.8804mm  max=1.3691mm  final=0.8271mm  solve_mean=0.031ms
N= 2   rms=0.2512mm  max=0.6092mm  final=0.0402mm  solve_mean=0.147ms
N= 3   rms=0.1080mm  max=0.3143mm  final=0.0078mm  solve_mean=0.219ms
N= 5   rms=0.0892mm  max=0.2471mm  final=0.0010mm  solve_mean=0.520ms
N=10   rms=0.0891mm  max=0.2446mm  final=0.0009mm  solve_mean=1.646ms
N=15   rms=0.0891mm  max=0.2446mm  final=0.0009mm  solve_mean=5.757ms

reference: naive_inverse_jacobian_ltv   rms=0.0135mm  max=0.0739mm  final=0.0002mm  solve_mean=0.126ms
```
RMS tracking error is essentially converged by $N=5$ (0.089mm) and is
**bit-for-bit identical** from $N=10$ to $N=15$ (0.0891mm both) — i.e. $N=15$
sits deep inside the converged plateau, not at a marginal cutoff. Solve time
grows roughly linearly with $N$ (0.03ms→5.8ms) but even at $N=15$ is far
below the 70–100ms control-loop budget.

### S12.2 Horizon sweep on both live plan shapes (rectangle + triangle)

```
=== rectangle ===
  N= 1  rms=0.5619mm  max=1.2215mm  %D_neg=71.9%  cos_median=-0.195
  N= 3  rms=0.0768mm  max=0.2178mm  %D_neg=28.1%  cos_median=+0.765
  N= 5  rms=0.0517mm  max=0.2145mm  %D_neg=34.5%  cos_median=+0.760
  N=10  rms=0.0500mm  max=0.2145mm  %D_neg=32.4%  cos_median=+0.774
  N=15  rms=0.0500mm  max=0.2145mm  %D_neg=32.4%  cos_median=+0.774

=== triangle ===
  N= 1  rms=0.0725mm  max=0.1150mm  %D_neg=54.6%  cos_median=-0.104
  N= 3  rms=0.0100mm  max=0.0783mm  %D_neg=52.9%  cos_median=+0.698
  N= 5  rms=0.0100mm  max=0.0778mm  %D_neg=57.1%  cos_median=+0.690
  N=10  rms=0.0100mm  max=0.0778mm  %D_neg=56.3%  cos_median=+0.692
  N=15  rms=0.0100mm  max=0.0778mm  %D_neg=56.3%  cos_median=+0.692
```
($D_k$: a per-tick directional/damping diagnostic from the original script;
`cos_median`: median cosine between successive correction directions —
strongly negative at $N=1$ on both shapes, indicating $N=1$ oscillates/
overcorrects, and positive from $N=3$ on, indicating coherent, non-oscillatory
corrections once the horizon is long enough to look past the immediate
step.) On **both** shapes independently, RMS/max converge by $N=5$ and are
bit-identical from $N=10\to15$. $N=1$ is qualitatively different and worse
on both metrics and both shapes (negative correction-direction cosine — the
controller fights itself tick to tick) — this is the clearest evidence in
this whole document that $N$ cannot be pushed short for cost/latency
without a real behavioral regression, and that $N=15$ is comfortably past
the point of diminishing returns rather than an arbitrary large choice.

**Caveat repeated for clarity:** both S12.1 and S12.2 were run on the
predecessor undelayed LTV MPC, before the delay-augmented (13D state)
prediction model existed. They establish that this horizon-convergence
*shape* (plateau by $N\approx5$, flat by $N=10$) is a property of the
beam/robot dynamics and linearization error on these two plan shapes — a
reasonable basis for treating $N=15$ as safely inside the plateau — but they
are not a delay-aware-model-specific re-validation of $N=15$.

---

## S13. Index: which document section each table supports

| `CONTROLLER_JUSTIFICATION.md` section | supporting table(s) here |
|---|---|
| §2 delay model $d=2$ | S10.1–S10.2 |
| §3 disturbance persistence $\beta_d=1$ | S11.1–S11.3 |
| §4 prediction horizon $N=15$, $V_f=0$ | S12.1–S12.2 |
| §5 $Q_p$ unchanged | (see $E_{\rm model}$ note — not separately tabled; stayed 0.34–0.6mm in every replay across §S2–S8) |
| §6.1 MT rejected | S2.1–S2.7 |
| §6.2 $Q_\gamma$ sweep | S3.1–S3.2, S4.1–S4.2 |
| §6.3 $\gamma=0$/R700 frozen | S5.1–S5.4, S7.1–S7.2 |
| §6.4 $Q_N$ necessity | S8.1–S8.2 |
| §7 $R$=700$R_0$ | S5.2, S5.3 |
| §8 $R_d$ unchanged, scale pinned | S4.1 (mild-effect evidence) |
| §9 insertion authority kept | S6.1–S6.7 |
| §11 γ=1↔M0 regression guard | S1.1 |
| §12 process isolation | (see `worker_process.py` docstring; stress-test numbers not re-run here as they require live hardware) |
| worker startup verification | S1.3 |
