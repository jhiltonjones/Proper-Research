# Controller formulations — inverse-Jacobian (DLS) vs. MPC (condensed QP)

Both controllers act on the state **z** = (q₁,...,q₆, L) ∈ ℝ⁷ (six robot joints + beam
insertion length) and output a velocity command **v** = dz/dt ∈ ℝ⁷, applied each tick as
z_target = z_measured + Δt·**v**. Both use the same beam-tip Jacobian J(z) ∈ ℝ³ˣ⁷ =
∂p_tip/∂z, where p_tip ∈ ℝ³ is the measured beam-tip position.

---

## 1. Inverse-Jacobian (damped least squares + nullspace regulation)

A single-step law, re-solved fresh every tick — no horizon, no prediction.

**Tip-position error at the next reference sample:**

&nbsp;&nbsp;&nbsp;&nbsp;**e** = p_desired(k+1) − p_measured

**Damped pseudo-inverse** (Levenberg parameter λ = `damping`):

&nbsp;&nbsp;&nbsp;&nbsp;J⁺ = Jᵀ(JJᵀ + λ²I)⁻¹

**Task-space (row-space) velocity** (gain k_p = `position_gain`, default 0.6):

&nbsp;&nbsp;&nbsp;&nbsp;**v**_task = J⁺ · (k_p **e** / Δt)

**Null-space projector** (exact, since J⁺J is idempotent):

&nbsp;&nbsp;&nbsp;&nbsp;P_null = I₇ − J⁺J

**Null-space velocity** (gain k_n = `nullspace_gain`, default 1.0; pulls toward reference
configuration z_ref, along the 7−3 = 4 directions that don't move the tip):

&nbsp;&nbsp;&nbsp;&nbsp;**v**_null = P_null · (k_n (z_ref(k+1) − z_measured) / Δt)

**Feedforward term** (only the null-space-consistent part of the reference velocity, since
**v**_task already carries the tip motion the reference asks for):

&nbsp;&nbsp;&nbsp;&nbsp;**v**_ff = P_null · **v**_reference

**Total command**, then clipped to velocity/acceleration/state-box limits:

&nbsp;&nbsp;&nbsp;&nbsp;**v** = clip( **v**_task + **v**_null + **v**_ff )

**Key structural fact:** **v**_task ∈ row(J) and **v**_null, **v**_ff ∈ null(J) exactly, by
construction (P_null projects onto null(J) by definition). There is no free parameter that can
route command into the null space beyond the small, deliberate k_n term.

**Optional selective damping** (per-singular-value λ, off by default): with SVD J = UΣVᵀ,
Σ = diag(σ₁,σ₂,σ₃),

&nbsp;&nbsp;&nbsp;&nbsp;λᵢ² = λ² + g·wᵢ,&nbsp;&nbsp; wᵢ = 1/(σᵢ²+f²) − min_j 1/(σⱼ²+f²)

&nbsp;&nbsp;&nbsp;&nbsp;J⁺_sel = V·diag(σᵢ/(σᵢ²+λᵢ²))·Uᵀ

(g = `selective_damping_gain`, f = `selective_damping_floor`). wᵢ ≈ 0 for the well-conditioned
directions by construction (shifted by their own minimum), so only the near-null direction
picks up extra suppression.

---

## 2. MPC (condensed QP — shared by all three rungs: LTI / LTV-offline / LTV-SQP-online)

Optimizes a whole sequence **v**(0),...,**v**(N−1), N = 15 (this session's horizon), as one QP
every tick; only **v**(0) is applied (receding horizon).

**Predicted state** (linear):

&nbsp;&nbsp;&nbsp;&nbsp;z(j+1|k) = z(j|k) + Δt·**v**(j|k)

**Predicted tip position** (affine, using the scheduled/frozen/relinearized Jacobian J(k+j)):

&nbsp;&nbsp;&nbsp;&nbsp;p̂(j|k) = p_nominal(k+j) + J(k+j)·(z(j|k) − z_ref(k+j)) + d̂(k)

d̂(k) is an online-estimated plant/model mismatch offset.

**Cost, minimized over the whole horizon:**

&nbsp;&nbsp;&nbsp;&nbsp;J = Σ_{k=0}^{N−1} [ (z_k−z_ref,k)ᵀQ(z_k−z_ref,k) + (v_k−v_ref,k)ᵀR(v_k−v_ref,k) + Δv_kᵀR_d Δv_k + (p̂_k−p_desired,k)ᵀQ_p(p̂_k−p_desired,k) ] + terminal cost

with:

&nbsp;&nbsp;&nbsp;&nbsp;Q = w_state · diag(1/s_state²)&nbsp;&nbsp;&nbsp;&nbsp;— **this session: w_state = 0.001 (near-off)**

&nbsp;&nbsp;&nbsp;&nbsp;R = w_input · diag(1/s_input²)&nbsp;&nbsp;&nbsp;&nbsp;— isotropic, diagonal in raw joint/insertion coordinates

&nbsp;&nbsp;&nbsp;&nbsp;R_d = w_increment · diag(1/s_increment²)&nbsp;&nbsp;&nbsp;&nbsp;— isotropic, diagonal in raw joint/insertion coordinates

&nbsp;&nbsp;&nbsp;&nbsp;Q_p = w_position · diag(1/s_position²)&nbsp;&nbsp;&nbsp;&nbsp;— output (tip) tracking

**Critical asymmetry with inverse-Jacobian:** Q, R, R_d are all diagonal in the *raw* 7-D
coordinates — none of them know anything about J's singular directions. Only Q_p is
Jacobian-aware, and only indirectly (through p̂, which contains J). So null-space motion
(motion with zero effect on p̂) costs the same under R/R_d as row-space motion — nothing in
the default cost structurally suppresses it. It is bounded only by whatever the solver finds
optimal once Q_p's term is satisfied.

**Terminal cost:** either a fixed multiple of Q at k=N, or (`use_dare_terminal_cost=True`,
this session's setting) the solution P of the discrete algebraic Riccati equation for the
input-increment-augmented system

&nbsp;&nbsp;&nbsp;&nbsp;x_aug = [z−z_ref; v_prev−v_ref],&nbsp; A_aug = [[I, ΔtI],[0, I]],&nbsp; B_aug = [ΔtI; I]

&nbsp;&nbsp;&nbsp;&nbsp;Q_aug = [[Q + JᵀQ_pJ, 0],[0, R]],&nbsp; R_aug = R + R_d

evaluated per horizon-step along the reference schedule (LTV Riccati), or once (LTI).

**Optional directional damping** (off by default; tested this session at 0.1, 0.5, 2.0): for
each horizon step's *joint-only* 3×6 local Jacobian, SVD J₆ = UΣVᵀ, and

&nbsp;&nbsp;&nbsp;&nbsp;wᵢ = 1/(σᵢ²+f²) − min_j 1/(σⱼ²+f²),&nbsp;&nbsp; H_extra = g · V·diag(w)·Vᵀ

added to that step's 6×6 joint block of the Hessian. This is the direct analog of
inverse-Jacobian's selective damping, and the *only* Jacobian-direction-aware term in the
input cost.

**Solved as one condensed QP** in the stacked decision variable **v** = [**v**(0);...;**v**(N−1)] ∈ ℝ⁷ᴺ:

&nbsp;&nbsp;&nbsp;&nbsp;minimize 0.5 **v**ᵀH**v** + qᵀ**v**&nbsp;&nbsp; subject to input/state/increment box constraints

&nbsp;&nbsp;&nbsp;&nbsp;H = 2(SᵀQ̄S + R̄ + DᵀR̄_dD [+ Σ directional-damping blocks])

S, D are condensed block matrices (S: block lower-triangular ΔtI, turning input sequence into
predicted states; D: block first-difference, turning input sequence into increments); Q̄, R̄,
R̄_d are the block-diagonal stackings of Q, R, R_d, Q_p over the horizon.

---

## Side-by-side

| | Inverse-Jacobian | MPC |
|---|---|---|
| Horizon | 1 step | N = 15, receding |
| Row/null split | **Structural** (P_null exact by construction) | **Emergent** (only Q_p distinguishes row from null; R, R_d don't) |
| Native Jacobian-direction awareness | Built into J⁺'s gain σ/(σ²+λ²) | Absent by default; only via optional directional damping |
| Row-space aggressiveness knob | k_p (`position_gain`) | No single equivalent — emergent from horizon-wide cost trade-off |
| This session, empirical | row-space ≈0.05–0.06 rad/s, null-space ≈0.5–1% | row-space ≈0.09–0.12 rad/s (≈2×), null-space ≈45–61% (default), 46% at directional_damping=0.5 |

The asymmetry is structural, not a tuning oversight: inverse-Jacobian's P_null = I−J⁺J is an
exact algebraic projector, so null-space command can only ever equal the small, deliberate k_n
term. MPC's null-space usage is whatever remains after Q_p's cost is satisfied, regulated only
by the isotropic R/R_d unless directional damping is turned on and pushed hard enough — this
session, g=0.1 (day2's validated value on a different shape/calibration) barely moved it
(55.9%→54.3%), g=0.5 moved it substantially (→46.4%), confirming the mechanism works but needs
recalibrating per-shape/per-calibration rather than reusing a fixed prior value.
