# Continuous centreline referencing and contouring MPC objective

## 1. Core idea

The controller is no longer asked to hit a sequence of discrete centreline nodes exactly. Instead, it treats the lumen centreline as a continuous curve parameterised by arc length and penalises two different components of tip-position error:

- **contour error**: displacement normal to the centreline;
- **lag error**: displacement along the centreline tangent.

The contour error receives the larger weight. This makes the tip follow the path without forcing it to hit every sampled point at exactly the correct prediction stage.

A separate distinction is essential:

- the **path tangent** is geometry used to split the position error;
- the **beam tip tangent** is an optional controller output used for safety or optional tangent tracking.

The controller can therefore use a path tangent while still modelling only the tip position (`n_out=3`). With `n_out=6`, it can also model the beam tip tangent but assign it zero tracking weight.

---

## 2. Centreline parameterisation by arc length

Let the sampled centreline be

\[
C_0,C_1,\ldots,C_{M-1},\qquad C_i\in\mathbb R^3.
\]

For segment \(i\), define

\[
d_i=C_{i+1}-C_i,
\qquad
\ell_i=\|d_i\|,
\qquad
\hat t_i=\frac{d_i}{\ell_i}.
\]

The cumulative arc-length coordinate is

\[
s_0=0,
\qquad
s_{i+1}=s_i+\ell_i.
\]

This changes the reference coordinate from a point index to a physical distance in metres. Parameters such as lookahead, stride and search window are therefore independent of centreline sampling density.

Duplicate consecutive centreline points must be rejected because they give \(\ell_i=0\), making the tangent and projection undefined.

---

## 3. Continuous projection of the measured tip

For a tip position \(p\) and a candidate segment from \(A=C_i\) to \(B=C_{i+1}\), calculate

\[
\alpha_i
=
\operatorname{clip}
\left(
\frac{(p-A)^T(B-A)}{\|B-A\|^2},
0,
1
\right).
\]

The closest point on that finite segment is

\[
\Pi_i(p)=A+\alpha_i(B-A).
\]

The corresponding continuous path coordinate is

\[
s_i^{\mathrm{proj}}
=
s_i+\alpha_i\ell_i.
\]

The selected segment is the candidate with minimum \(\|p-\Pi_i(p)\|\).

This is better than choosing the nearest centreline node because the result changes continuously as the tip moves along a segment. Node-based referencing can jump abruptly from one sample to the next.

---

## 4. Global initial projection and local subsequent projection

At initialisation, all path segments are candidates. This allows the simulation or experiment to begin anywhere along the lumen.

After initialisation, candidates are restricted to a window around the previous path coordinate:

\[
[s_{\mathrm{previous}}-w_{\mathrm{back}},
 s_{\mathrm{previous}}+w_{\mathrm{forward}}].
\]

This local search prevents a spatially nearby but topologically distant branch from being selected at a self-crossing or tight fold.

Typical parameters are:

```python
ref_search_backward_m = 5.0e-4
ref_search_forward_m = 5.0e-3
```

The forward window is normally larger because forward progress is expected. The backward window only gives the projector enough tolerance to accommodate measurement/model error.

---

## 5. Monotonic, rate-limited progress state

The geometric projection is not used directly as the controller's progress state. It is filtered:

\[
s_{\mathrm{progress}}
=
\operatorname{clip}
\left(
s_{\mathrm{projected}},
s_{\mathrm{lower}},
s_{\mathrm{upper}}
\right).
\]

With backward motion disabled,

\[
s_{\mathrm{lower}}=s_{\mathrm{previous}},
\qquad
s_{\mathrm{upper}}
=s_{\mathrm{previous}}+\Delta s_{\max}.
\]

Thus the reference cannot move backwards and cannot jump forwards by more than `ref_max_progress_per_step_m` during one controller frame.

This state has two purposes:

1. it prevents branch switching and reference reversal;
2. it prevents one noisy projection from moving the complete horizon far ahead.

The cap must still be large enough to accommodate the maximum physically possible path progress per controller frame. An excessively small cap makes the reference lag artificially behind the robot.

---

## 6. Building the prediction-horizon reference

Once \(s_{\mathrm{progress}}\) is known, the stage coordinates are

\[
s_k^{\mathrm{ref}}
=
s_{\mathrm{progress}}
+s_{\mathrm{lookahead}}
+k\,s_{\mathrm{stride}},
\qquad k=0,\ldots,N_p-1.
\]

They are clipped to the total path length.

- `ref_lookahead_m` determines how far ahead of the current progress anchor the first target lies.
- `ref_stride_m` determines the physical spacing between successive horizon targets.
- `Np * ref_stride_m` roughly controls how much path is represented by the horizon.

The centreline position at each \(s_k^{\mathrm{ref}}\) is obtained by interpolation within its segment:

\[
C(s)=(1-\alpha)C_i+\alpha C_{i+1}.
\]

The code also interpolates and normalises path tangents. In the latest patch, node tangents are length-weighted averages of adjacent segment tangents, which reduces direction jumps at polyline vertices while leaving the centreline positions unchanged.

The reference object contains

```text
s_progress
s_ref
C_ref
T_ref
segment_indices
projection
```

The old integer indices are retained only for compatibility and logging in contouring mode.

---

## 7. Why the reference is frozen during an SQP solve

An SQP controller builds several QPs around successive control guesses while the measured state \(x_0\) remains fixed.

The geometric reference must therefore be computed once before the SQP iterations and reused for every QP in that solve. Otherwise both the model linearisation and the target move simultaneously, making convergence difficult to interpret and often producing oscillation.

The implementation stores

```python
self.active_path_reference
self._active_path_reference_x0_key
```

and reuses the reference whenever the same measured-state key is seen. A new reference is built only on the next controller frame, when a new measured state is supplied.

---

## 8. Position error decomposition

At prediction stage \(k\), let

\[
e_k=p_k-C_k^{\mathrm{ref}},
\]

where \(p_k\) is the predicted tip position and \(t_k\) is the unit path tangent.

Define the tangent projector

\[
P_{\parallel,k}=t_kt_k^T
\]

and the normal-plane projector

\[
P_{\perp,k}=I-t_kt_k^T.
\]

Then

\[
e_{\parallel,k}=P_{\parallel,k}e_k,
\qquad
 e_{\perp,k}=P_{\perp,k}e_k.
\]

The scalar signed lag error used in diagnostics is

\[
\ell_k=t_k^Te_k,
\]

and

\[
e_{\parallel,k}=t_k\ell_k.
\]

The contour-error magnitude is

\[
c_k=\|e_{\perp,k}\|.
\]

Interpretation:

- contour error says how far the tip is from the centreline;
- lag error says whether the tip is ahead of or behind the selected path station.

---

## 9. Stage tracking objective

The position stage cost is

\[
J_{p,k}
=
q_c\,e_k^TP_{\perp,k}^TWP_{\perp,k}e_k
+
q_l\,e_k^TP_{\parallel,k}^TWP_{\parallel,k}e_k,
\]

where

- \(q_c\) is `q_contour`;
- \(q_l\) is `q_lag`;
- \(W=\operatorname{diag}(w_x,w_y,w_z)\) comes from `contouring_axis_weights`.

Therefore the position weight matrix is

\[
Q_{p,k}
=
q_cP_{\perp,k}^TWP_{\perp,k}
+
q_lP_{\parallel,k}^TWP_{\parallel,k}.
\]

With \(W=I\), the projectors are symmetric and idempotent, giving

\[
J_{p,k}=q_c\|e_{\perp,k}\|^2+q_l\|e_{\parallel,k}\|^2.
\]

Normally choose

\[
q_c\gg q_l.
\]

This says: staying close to the centreline is more important than arriving at an exact arc-length station at an exact horizon stage.

If \(q_l\) is too large, the formulation approaches time-indexed waypoint tracking and can again command aggressive catch-up motion. If \(q_l=0\), the controller may have little incentive to progress when effort is penalised, so either a small positive lag weight, a progress reward or a progress constraint is useful.

---

## 10. Horizon weighting

Each stage matrix is multiplied by `ref_stage_weights[k]`:

\[
\widetilde Q_k=w_kQ_k.
\]

These weights control the relative importance of near-term and terminal predictions.

- decreasing weights emphasise immediate, more reliable predictions;
- increasing weights emphasise the end of the horizon;
- uniform weights are the simplest baseline.

Because nonlinear model error usually grows along the horizon, a mild decrease is often defensible. The choice must be kept identical between contact and no-contact Jacobian experiments.

---

## 11. Optional beam tip-tangent tracking

The geometric path tangent \(T_k^{\mathrm{ref}}\) is always available to the path objective. This does not imply that the beam tip tangent is being controlled.

When `n_out=6`, the controller output is

\[
y_k=
\begin{bmatrix}
p_k\\t_k^{\mathrm{tip}}
\end{bmatrix}.
\]

The stacked reference also contains

\[
y_k^{\mathrm{ref}}=
\begin{bmatrix}
C_k^{\mathrm{ref}}\\T_k^{\mathrm{ref}}
\end{bmatrix}.
\]

However, the tangent objective block is taken from

```python
self.Q[3:6, 3:6]
```

so with

```python
w_tracking = (
    1000.0, 1000.0, 0.0,
    0.0, 0.0, 0.0,
)
```

the tangent block is exactly zero. The tangent is modelled but not tracked.

This still permits an independent hard tangent-angle safety constraint because a constraint uses the tangent rows of the prediction matrix directly; it does not require a non-zero tracking weight.

---

## 12. Affine SQP prediction used by the QP

Around a current SQP control guess \(U^{g}\), the nonlinear predicted output is linearised as

\[
X(U)
\approx
X_{\mathrm{nom}}
+M_c(U-U^g).
\]

Equivalently,

\[
X(U)\approx X_{\mathrm{aff}}+M_cU,
\]

where

\[
X_{\mathrm{aff}}=X_{\mathrm{nom}}-M_cU^g.
\]

This affine matching is important because the linear model reproduces the nonlinear nominal rollout exactly at \(U=U^g\).

The stacked tracking cost is

\[
J_{\mathrm{track}}(U)
=
(X_{\mathrm{aff}}+M_cU-X_{\mathrm{ref}})^T
\widetilde Q
(X_{\mathrm{aff}}+M_cU-X_{\mathrm{ref}}).
\]

OSQP uses

\[
\frac12U^THU+f^TU.
\]

Expanding the tracking term gives

\[
H_{\mathrm{track}}=2M_c^T\widetilde QM_c,
\]

\[
f_{\mathrm{track}}
=2M_c^T\widetilde Q(X_{\mathrm{aff}}-X_{\mathrm{ref}}).
\]

This is why the new objective required only a new `X_ref` and `Qtil`; the existing prediction and QP machinery remained valid.

---

## 13. Full objective

The implemented objective is conceptually

\[
J(U)=J_{\mathrm{track}}(U)
+J_{\mathrm{effort}}(U)
+J_{\mathrm{smooth}}(U)
+J_{\mathrm{progress}}(U)
+J_{\mathrm{reg}}(U).
\]

### Control effort

\[
J_{\mathrm{effort}}
=\sum_{k=0}^{N_p-1}u_k^TRu_k.
\]

This discourages unnecessarily large magnet and insertion commands.

### Control smoothness

Let

\[
\Delta u_0=u_0-u_{\mathrm{previous}},
\qquad
\Delta u_k=u_k-u_{k-1}.
\]

Then

\[
J_{\mathrm{smooth}}
=\sum_k\Delta u_k^TR_d\Delta u_k.
\]

The code represents this with a difference matrix \(D\), so

\[
\Delta U=DU-c_{\mathrm{previous}}.
\]

Its QP contribution is

\[
H_{\mathrm{smooth}}=2D^TR_dD,
\qquad
f_{\mathrm{smooth}}=-2D^TR_dc_{\mathrm{previous}}.
\]

### Regularisation

A small term

\[
\epsilon\|U\|^2
\]

is added through `qp_reg`. It improves numerical conditioning but should be much smaller than meaningful control penalties.

---

## 14. Optional progress model

A small lag weight may not guarantee forward progress because doing nothing can minimise effort while remaining near the path.

The terminal path coordinate is approximated locally by

\[
\hat s_N(U)
=
s_N^{\mathrm{ref}}
+T_N^T\left(p_N(U)-C_N^{\mathrm{ref}}\right).
\]

Because the terminal position is affine in \(U\), this becomes

\[
\hat s_N(U)=b_s+A_sU.
\]

Two optional uses are implemented:

1. **progress reward**
   \[
   -q_s\hat s_N(U),
   \]
   which adds only a linear objective term;
2. **hard progress constraint**
   \[
   \hat s_N(U)\ge s_{\mathrm{progress}}+\Delta s_{\mathrm{request}}.
   \]

The hard form is what can make a QP formally infeasible when safe progress is impossible under the model and constraints. It should be enabled only after the basic contouring reference is stable.

---

## 15. Implementation map

### `simulation_controller/reference.py`

Responsible for:

- mode dispatch: `point` versus `contouring`;
- path geometry and arc length;
- tip-to-segment projection;
- local search windows;
- monotonic/rate-limited progress state;
- horizon positions and tangents;
- reference state reset;
- freezing the active reference for one controller solve.

### `simulation_controller/qp_builder.py`

Responsible for:

- building `X_ref`;
- constructing the contour/lag stage matrices;
- expanding the tracking term into `H_track` and `f_track`;
- adding effort, smoothness and regularisation;
- optional progress reward/constraint;
- optional tangent tracking;
- path diagnostics.

### `controller.py` and `controller_optimized.py`

Responsible for:

- calling the reference selector once per measured controller state;
- keeping the reference fixed during SQP;
- resetting reference state when the controller is reset;
- solving the same mathematical control problem in debugging and optimized implementations.

### Controller factories

Responsible for assigning:

```text
reference_mode
q_contour
q_lag
contouring_axis_weights
ref_lookahead_m
ref_stride_m
ref_search_backward_m
ref_search_forward_m
ref_max_progress_per_step_m
progress_request_m
progress_reward_weight
enable_hard_progress_constraint
```

and, when `n_out=6`, the position/tangent output weights.

### Magnetic-beam model files

The optimized energy and equilibrium solver do not define the path objective. They supply the nonlinear forward model and implicit Jacobian. They only needed changes for optional six-output tip-position/tip-tangent modelling, not for the continuous position reference itself.

---

## 16. Minimal pseudocode implementation

```python
# Once per measured controller state
geometry = prepare_path_geometry(lumen_C)
projection = project_tip_locally(
    tip=measured_tip,
    previous_s=path_progress_s,
    geometry=geometry,
)

path_progress_s = clip_progress(
    projection.s,
    previous_s=path_progress_s,
    max_advance=ref_max_progress_per_step_m,
    allow_backward=False,
)

s_ref = path_progress_s + ref_lookahead_m \
        + ref_stride_m * np.arange(Np)
C_ref, T_ref = interpolate_path(geometry, s_ref)
active_path_reference = {"s_ref": s_ref, "C_ref": C_ref, "T_ref": T_ref}

# Reused in every QP of the current SQP solve
X_nom, Mc = build_local_prediction(U_guess)
X_aff = X_nom - Mc @ U_guess

for k in range(Np):
    t = T_ref[k]
    P_parallel = np.outer(t, t)
    P_perp = np.eye(3) - P_parallel

    Q_position = (
        q_contour * P_perp.T @ W @ P_perp
        + q_lag * P_parallel.T @ W @ P_parallel
    )

    Qk = np.zeros((n_out, n_out))
    Qk[:3, :3] = stage_weight[k] * Q_position

    if n_out >= 6:
        Qk[3:6, 3:6] = stage_weight[k] * Q_base[3:6, 3:6]

    place_block(Qtil, k, Qk)

H_track = 2 * Mc.T @ Qtil @ Mc
f_track = 2 * Mc.T @ Qtil @ (X_aff - X_ref)
```

---

## 17. Tuning sequence

A defensible tuning order is:

1. Disable hard progress and tangent-angle constraints.
2. Use `n_out=3` or `n_out=6` with zero tangent weights.
3. Set `contouring_axis_weights=(1,1,1)`.
4. Choose physical lookahead and stride from expected speed and sample period.
5. Set `q_lag` around 1–10% of `q_contour`.
6. Increase `q_contour` until centreline error is acceptable without persistent control saturation.
7. Add `Rd` if commands oscillate from one stage/frame to the next.
8. Add a small progress reward only if the controller stalls.
9. Enable hard progress for the explicit safe-progress feasibility experiment.
10. Enable the hard tangent-angle constraint only when the six-output tangent Jacobian is validated.

Example starting point:

```python
reference_mode = "contouring"
q_contour = 1000.0
q_lag = 50.0
contouring_axis_weights = np.ones(3)
ref_lookahead_m = 5.0e-4
ref_stride_m = 5.0e-4
ref_search_backward_m = 5.0e-4
ref_search_forward_m = 5.0e-3
ref_max_progress_per_step_m = 1.0e-3
progress_reward_weight = 0.0
enable_hard_progress_constraint = False
```

These are starting values, not universal physical constants.

---

## 18. Diagnostics to inspect

The most useful signals are:

```text
path_progress_current_m
path_projection_distance_m
path_ref_s
contour_error_nominal_rms_m
contour_error_nominal_max_m
contour_error_nominal_terminal_m
lag_error_nominal_rms_m
lag_error_nominal_terminal_m
progress_nominal_terminal_m
progress_nominal_shortfall_m
constraint_issue_guess
cond_H_mpc
```

Interpretation:

- large projection distance: the measured tip is far from the centreline or the local search window is wrong;
- discontinuous `path_progress_current_m`: progress-state management is being overwritten elsewhere;
- low contour error but large lag error: the tip is near the path but behind/ahead of the selected station;
- oscillatory signed lag error: lag weight, lookahead or stride may be too aggressive;
- persistent saturation: the target horizon or tracking weight is demanding more than the actuator model can provide;
- large condition number: weights may differ by too many orders of magnitude or the predicted control directions are weak.

---

## 19. Common mistakes

1. **Confusing path tangent with beam tip tangent.** The former is required for contouring error; the latter is optional output state.
2. **Using global nearest projection every frame.** This permits branch switching.
3. **Using node index as distance.** Behaviour then changes when the centreline is resampled.
4. **Rebuilding the reference inside every SQP iteration.** This creates a moving objective.
5. **Using a large lag weight.** This recreates exact time-indexed waypoint tracking.
6. **Setting lag weight to zero with no progress incentive.** The minimum-effort solution may stall.
7. **Overwriting `path_progress_s` from a separate global-nearest calculation in the runner.** The controller must own this monotonic state.
8. **Changing the reference between contact/no-contact Jacobian experiments.** That confounds the model comparison.
9. **Assuming zero tangent tracking weight disables the tangent safety constraint.** Objective weights and hard constraints are separate.
10. **Claiming infeasibility from poor tracking alone.** Formal QP infeasibility requires conflicting hard constraints, such as minimum progress plus safety/actuation bounds.

---

## 20. Reimplementation checklist

- [ ] Validate the centreline and remove duplicate consecutive points.
- [ ] Build cumulative arc length in metres.
- [ ] Project to line segments, not only nodes.
- [ ] Use global projection only for initialisation.
- [ ] Restrict later projection to a local arc-length window.
- [ ] Maintain monotonic, rate-limited progress state.
- [ ] Generate horizon stations with metric lookahead and stride.
- [ ] Interpolate both positions and unit tangents.
- [ ] Freeze one reference throughout each SQP solve.
- [ ] Split position error with tangent/normal projectors.
- [ ] Keep contour weight larger than lag weight.
- [ ] Expand the affine tracking cost correctly into `H` and `f`.
- [ ] Add effort, smoothness and regularisation separately.
- [ ] Keep tangent tracking optional through zero/non-zero tangent weights.
- [ ] Keep progress constraints separate from the soft tracking objective.
- [ ] Log projection, progress, contour, lag and conditioning diagnostics.
- [ ] Use identical reference settings for contact and no-contact Jacobian comparisons.
