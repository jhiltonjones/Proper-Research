"""Target-consistent configuration shift for delay-aware MPC (2026-09-19).

Motivated by a real-data finding from `close_loop_logs/myrun/
mpc_delay_aware_accumC_20260918T165039Z` (a genuinely good live run,
final_error_mm=1.138, max_error_mm=2.514): decomposing the SAME run's own
logged predictions into

    E_model,+j = RMS||p_meas,k+j - phat_k+j|k||   (is the model accurate?)
    E_chosen,+j = RMS||phat_k+j|k - p_des,k+j||    (is the CHOSEN trajectory good?)

gives E_model,+3=0.435mm but E_chosen,+3=1.321mm -- the physical prediction
is accurate, but the optimizer's own chosen future, judged against ITS OWN
model, tolerates ~3x more task error than the model's accuracy would
require. The cost-gradient decomposition earlier in this investigation
found cos(g_Q,0, g_Qp,0) ~= -1 (Q and Qp almost exactly opposing) --
consistent with the state-tracking cost (centred at z_ref) and the task
cost (driving toward p_des) fighting each other, with the state cost
"winning" enough to leave a persistent ~1mm bias (E_bias~=1.06mm on this
same run, E_var~=0.82mm -- the bias, not the noise, dominates).

This module implements the fix WITHOUT touching the plant model: given the
SAME already-validated disturbance estimate d_k (unchanged, beta_d=1,
still assumed persistent over the horizon) and the SAME already-validated
Jacobian schedule, compute -- for each horizon stage j -- the smallest
(in a physically-normalized weighted-least-norm sense) configuration
displacement from z_ref,j that would CANCEL the disturbance in the
CONTROLLED task subspace (the beam plane, exactly the same 2D subspace
`close_loop_path_follow.py`'s own `project_error_to_beam_plane` already
uses -- this reuses that project's `beam_axial_axis_R`/
`beam_plane_normal_axis_R` convention so the target calculator solves
literally the same task the feedback loop is already scored against).

That displaced point z^t_j becomes a new Q-cost CENTER only:

    ||z_phys,j - z_ref,j||_Q^2   -->   ||z_phys,j - z^t_j||_Q^2

The physical PREDICTION equation (E_p/S_p, the beam Jacobian linearization,
d_k's role in `constant_error`) is entirely unchanged -- it was linearized
about z_ref and stays linearized there. Only the state-tracking cost's
center moves; H_Q is untouched (shifting a quadratic's center only changes
its linear term), and no hard constraint is touched.

Math (matches the exact formulation specified alongside this module):

    C in R^{2x3}          orthonormal basis for the beam plane (rows),
                           built from the SAME beam_axial_axis_R (one
                           in-plane row, unit-norm by construction) and
                           beam_plane_normal_axis_R (its cross product
                           with the axial axis gives the second in-plane
                           row) this project already uses for
                           project_error_to_beam_plane -- NOT independently
                           chosen, so the target calculator solves exactly
                           the task the live feedback loop is scored on.

    J^t_j = C @ J_{k+j}   (2x7)
    d^t_k = C @ d_k       (2,)

    W = diag(1/s_q1^2, ..., 1/s_q6^2, 1/s_L^2)   (7x7, SAME scales as Q)

    delta_z*_j = argmin (1/2) delta_z^T W delta_z
                 s.t.    J^t_j @ delta_z = -d^t_k

    closed form (J^t_j W^{-1} J^t_j^T is 2x2, generically invertible since
    C has full row rank 2 and J_j has rank >= 2 in-plane on this system):

        delta_z*_j = -W^{-1} @ J^t_j^T @ inv(J^t_j @ W^{-1} @ J^t_j^T) @ d^t_k

    z^t_{k+j} = z_ref,k+j + delta_z*_j

First-experiment scope deliberately does NOT implement the bound-
constrained lexicographic fallback the design discussion also specified:
z^t is a COST CENTER, not a hard constraint -- the existing joint-position/
velocity/rate boxes in `_constraint_bounds_exec` still enforce every real
safety limit regardless of what z^t says, so an individually "infeasible"
unconstrained z^t is not unsafe, just a strong pull the QP will trade off
against Q/Qp/R/Rd as usual. Revisit only if live testing shows this
matters in practice.
"""
from __future__ import annotations

import numpy as np

Array = np.ndarray


def build_beam_plane_projection(
    *, axial_axis_R: Array, normal_axis_R: Array,
) -> Array:
    """(2,3) orthonormal projection C: row 0 is the (unit) axial axis, row 1
    is normal x axial (completing a right-handed in-plane basis). Matches
    `close_loop_path_follow.py`'s own beam-plane convention exactly --
    that code subtracts the component along `normal_axis_R` from the raw
    3D error; C's rows span precisely the complementary 2D subspace."""
    t1 = np.asarray(axial_axis_R, dtype=float).reshape(3)
    n = np.asarray(normal_axis_R, dtype=float).reshape(3)
    t1 = t1 / np.linalg.norm(t1)
    n = n / np.linalg.norm(n)
    if abs(float(np.dot(t1, n))) > 1e-9:
        raise ValueError(
            f"axial_axis_R and normal_axis_R must be orthogonal, got "
            f"dot={float(np.dot(t1, n)):.3e}"
        )
    t2 = np.cross(n, t1)
    t2 = t2 / np.linalg.norm(t2)
    return np.stack([t1, t2], axis=0)  # (2,3)


def compute_target_shift(
    *, J_stage: Array, d_k: Array, C: Array, w_diag: Array,
) -> tuple[Array, Array]:
    """One horizon stage's task-consistent configuration shift.

    J_stage: (3,7) beam Jacobian at this stage's reference index.
    d_k: (3,) disturbance estimate (already beta_d-scaled by the caller,
        matching `_beam_prediction_terms_exec`'s own `beta_d * estimated`).
    C: (2,3) from `build_beam_plane_projection`.
    w_diag: (7,) diagonal of W = diag(1/s_q^2, ..., 1/s_L^2).

    Returns (delta_z (7,), residual (2,) = J^t @ delta_z + d^t -- should be
    ~0 whenever J^t @ W^{-1} @ J^t.T is well-conditioned, which is the
    normal case here since C has full row rank 2).
    """
    J_stage = np.asarray(J_stage, dtype=float).reshape(3, 7)
    d_k = np.asarray(d_k, dtype=float).reshape(3)
    w_inv = 1.0 / np.asarray(w_diag, dtype=float).reshape(7)

    Jt = C @ J_stage  # (2,7)
    dt = C @ d_k  # (2,)

    # W^{-1} J^tT is (7,2); J^t W^{-1} J^tT is (2,2)
    Winv_JtT = w_inv[:, None] * Jt.T  # (7,2), W^{-1} diagonal broadcast
    M = Jt @ Winv_JtT  # (2,2)
    lam = np.linalg.solve(M, -dt)  # (2,)
    delta_z = Winv_JtT @ lam  # (7,)

    residual = Jt @ delta_z + dt
    return delta_z, residual


def build_target_stack(
    *, state_reference: Array, J_schedule: Array, d_k: Array, C: Array, w_diag: Array,
) -> tuple[Array, Array]:
    """Vectorized over the N horizon stages.

    state_reference: (N,7) z_ref for stages 1..N.
    J_schedule: (N,3,7) beam Jacobian at each stage's reference index.
    d_k: (3,) disturbance estimate, ALREADY beta_d-scaled (constant over
        the horizon under beta_d=1 persistence, per this investigation's
        earlier disturbance-persistence result).

    Returns (z_target (N,7), residual (N,2)).
    """
    state_reference = np.asarray(state_reference, dtype=float).reshape(-1, 7)
    J_schedule = np.asarray(J_schedule, dtype=float).reshape(-1, 3, 7)
    N = state_reference.shape[0]
    assert J_schedule.shape[0] == N
    z_target = np.empty((N, 7), dtype=float)
    residual = np.empty((N, 2), dtype=float)
    for j in range(N):
        delta_z, r = compute_target_shift(
            J_stage=J_schedule[j], d_k=d_k, C=C, w_diag=w_diag,
        )
        z_target[j] = state_reference[j] + delta_z
        residual[j] = r
    return z_target, residual


def run_self_test() -> None:
    rng = np.random.default_rng(0)

    # --- 1: C matches the project's beam-plane convention exactly ---
    axial = np.array([-1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, -1.0])
    C = build_beam_plane_projection(axial_axis_R=axial, normal_axis_R=normal)
    assert np.allclose(C[0], axial), "row 0 must be the axial axis exactly"
    assert abs(np.dot(C[1], normal)) < 1e-12, "row 1 must be orthogonal to the plane normal"
    assert abs(np.dot(C[0], C[1])) < 1e-12, "C's rows must be orthogonal to each other"
    assert np.allclose(np.linalg.norm(C, axis=1), 1.0), "C's rows must be unit norm"
    print("[target_consistent self-test] 1/4 PASS: C matches the project's beam-plane "
          "convention (axial row exact, orthonormal, right-handed)")

    # --- 2: zero disturbance -> delta_z = 0 exactly ---
    J = rng.normal(size=(3, 7)) * 0.05
    w_diag = np.array([1.0 / (0.5e-2) ** 2] * 6 + [1.0 / 0.25e-3 ** 2])  # matches Q's own scales (rad, m)
    delta_z0, resid0 = compute_target_shift(J_stage=J, d_k=np.zeros(3), C=C, w_diag=w_diag)
    assert np.allclose(delta_z0, 0.0, atol=1e-12), "zero disturbance must give delta_z=0 exactly"
    assert np.allclose(resid0, 0.0, atol=1e-12)
    print("[target_consistent self-test] 2/4 PASS: zero disturbance -> delta_z=0 exactly")

    # --- 3: residual ~0 (exact cancellation in the controlled task) for a
    # generic disturbance, and the correction is minimum-W-norm ---
    d_k = rng.normal(size=3) * 1.0e-3
    delta_z, resid = compute_target_shift(J_stage=J, d_k=d_k, C=C, w_diag=w_diag)
    assert np.max(np.abs(resid)) < 1e-9, f"task residual should be ~0, got {resid}"
    # minimum-W-norm: delta_z must lie in the row space of W^{-1} J^tT, i.e.
    # W @ delta_z must lie in the row space of J^t (equivalently: any vector
    # orthogonal to J^t's rows, in the STANDARD inner product, is orthogonal
    # to W @ delta_z). Check by projecting W@delta_z onto span(J^t rows)
    # and confirming it's unchanged.
    Jt = C @ J
    W_delta = w_diag * delta_z
    proj = Jt.T @ np.linalg.lstsq(Jt @ Jt.T, Jt @ W_delta, rcond=None)[0]
    assert np.allclose(proj, W_delta, atol=1e-8), (
        "delta_z is not minimum-W-norm -- W @ delta_z should lie entirely "
        "in the row space of J^t"
    )
    print("[target_consistent self-test] 3/4 PASS: exact task cancellation "
          "(residual~0) and delta_z is minimum-W-norm")

    # --- 4: build_target_stack vectorization matches per-stage calls ---
    N = 6
    state_reference = rng.normal(size=(N, 7)) * 0.1
    J_schedule = rng.normal(size=(N, 3, 7)) * 0.05
    z_target, residual = build_target_stack(
        state_reference=state_reference, J_schedule=J_schedule, d_k=d_k, C=C, w_diag=w_diag,
    )
    for j in range(N):
        dz, r = compute_target_shift(J_stage=J_schedule[j], d_k=d_k, C=C, w_diag=w_diag)
        assert np.allclose(z_target[j], state_reference[j] + dz)
        assert np.allclose(residual[j], r)
    print("[target_consistent self-test] 4/4 PASS: build_target_stack matches "
          "per-stage compute_target_shift exactly")

    print("[target_consistent self-test] ALL PASS")


if __name__ == "__main__":
    run_self_test()
