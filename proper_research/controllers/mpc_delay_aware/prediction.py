"""Delay-augmented state prediction matrices for MPC (2026-09-18).

Standalone, independently-unit-tested core of the delay-aware MPC
formulation -- deliberately kept separate from
`simulate_time_parameterized_beam_output_mpc.BeamOutputTrackingMPC` (a
~1500-line class where `state` is currently used for two DIFFERENT roles
that the standard undelayed model happens to conflate: (1) the seed for
`E @ state` state-PREDICTION across the horizon, and (2) the MEASURED
state used to compute the disturbance/residual estimate `d_k`. Those two
roles are identical when q_phys == q_meas, which is exactly the assumption
this delay model removes -- so a safe integration needs each call site
audited for which role `state` is playing there, not a blanket swap. This
module gets the PREDICTION half exactly right and independently checkable
before that integration is attempted.

Physical model (validated: `hardware/online/rectangle_stage_a/README.md`,
d=2 samples, 5-fold LOO G_delay=0.945-0.986, unit-tested indexing --
a decision at tick k first affects q_phys at k+3, not k+1 or k+2):

    c_{j+1|k} = c_{j|k} + dt * u_{q,j|k}     (persistent command accumulator,
                                                 execution-C, unchanged)
    q_phys_{j|k} = c_{j-2|k}                  (delayed READ of the command
                                                 queue, j=1..N)
    L_{j|k} = L_k + dt * sum_{r=0}^{j-1} u_{L,r|k}   (insertion: UNDELAYED --
                                                        d=2 was identified
                                                        for the six UR
                                                        joints only, not the
                                                        advancer, see the
                                                        module-level note in
                                                        inverse_jacobian_2dof_delay_aware.py)

Written out for j=1,2,3 (delay_samples=2):

    q_phys_{1|k} = c_{-1|k} = q_cmd_{k-1}     (fully known, zero u-dependence)
    q_phys_{2|k} = c_{0|k}  = q_cmd_k         (fully known, zero u-dependence)
    q_phys_{3|k} = c_{0|k} + dt*u_{q,0|k}     (u_0 first appears HERE)

The "state" this module's E/S matrices are seeded from is therefore NOT the
7D measured/commanded state -- it's a 13D augmented vector

    z0_aug = [c_k (6), c_{k-1} (6), L_k (1)]

(the accumulator's OWN current and previous command, plus the current
insertion). `E` has shape (N*7, 13); `S` keeps the SAME shape the
undelayed model's S has (N*7, N*7) since it still maps the same N*7-long
decision vector to an N*7-long PHYSICAL state stack -- only its internal
sparsity pattern differs (shifted by delay_samples for the joint block).
"""
from __future__ import annotations

import numpy as np

Array = np.ndarray


def build_delay_prediction_matrices(
    *, N: int, dt: float, delay_samples: int = 2, n_joints: int = 6,
) -> tuple[Array, Array]:
    """Returns (E, S): E has shape (N*7, 13), S has shape (N*7, N*7).

    Stage/row index j runs 1..N (the usual MPC convention -- j=1 is the
    first PREDICTED stage, one tick after the current measurement); input
    index r runs 0..N-1 (u_{r|k}, r=0 is the decision applied at THIS
    tick). n = n_joints + 1 = 7 per stage (6 joints + insertion), matching
    the existing ConfigurationTrackingMPC/BeamOutputTrackingMPC convention.
    """
    d = int(delay_samples)
    if d > 2:
        raise ValueError(
            f"delay_samples={d}: z0_aug=[c_k, c_(k-1), L_k] only carries "
            f"one step of command history -- only delay_samples in {{0,1,2}} "
            f"is supported without extending z0_aug. This is deliberate, "
            f"not an oversight: d=2 is the only value validated so far."
        )
    n = n_joints + 1
    nu = N * n

    E = np.zeros((N * n, 2 * n_joints + 1), dtype=float)
    S = np.zeros((N * n, nu), dtype=float)

    for j in range(1, N + 1):
        row = (j - 1) * n
        # --- joint rows (delayed): q_phys_{j|k} = c_{j-d} (relative index,
        # c_0=c_k, c_i = c_k + dt*sum_{r=0}^{i-1}u_r for i>=1) ---
        rel = j - d
        if rel <= 0:
            # a PAST, already-known command -- z0_aug carries exactly one
            # step of history (c_{k-1}), so only rel in {0, -1} is reachable
            # (i.e. d in {0,1,2}), enforced above.
            if rel == 0:
                E[row:row + n_joints, 0:n_joints] = np.eye(n_joints)          # c_k
            else:
                E[row:row + n_joints, n_joints:2 * n_joints] = np.eye(n_joints)  # c_{k-1}
            # zero S contribution -- no u dependence yet
        else:
            # q_phys_{j|k} = c_k + dt * sum_{r=0}^{rel-1} u_{q,r}
            E[row:row + n_joints, 0:n_joints] = np.eye(n_joints)  # c_k (constant offset)
            last_r = rel - 1  # inclusive, 0-indexed
            for r in range(0, last_r + 1):
                col = r * n
                S[row:row + n_joints, col:col + n_joints] = dt * np.eye(n_joints)

        # --- insertion row (undelayed, standard cumsum from L_k) ---
        E[row + n_joints, 2 * n_joints] = 1.0  # picks out L_k
        for r in range(0, j):
            col = r * n + n_joints
            S[row + n_joints, col] = dt

    return E, S


def predict_z_phys(
    *, E: Array, S: Array, z0_aug: Array, v: Array, N: int, n: int = 7,
) -> Array:
    """(N, n) predicted physical-state stack, stage 1..N."""
    z0_aug = np.asarray(z0_aug, dtype=float).reshape(-1)
    v = np.asarray(v, dtype=float).reshape(-1)
    return (E @ z0_aug + S @ v).reshape(N, n)


def run_self_test() -> None:
    """The four regression tests specified before any QP integration is
    attempted. Run via `python -m proper_research.controllers.mpc_delay_aware.prediction`.

    2026-09-18: test 1, as originally written, caught a real off-by-one bug
    in the first draft of `build_delay_prediction_matrices` (`last_r` was
    `j - d - 2` instead of `j - d - 1`, which silently gave q_phys_3 ZERO
    u_0-sensitivity instead of dt*I -- i.e. the delay would have been
    modelled as 3 samples, not the validated 2, had this gone untested).
    Left as a standing regression test, not a one-off check.
    """
    N, dt, n_joints = 6, 0.1, 6
    rng = np.random.default_rng(0)

    # --- 1: partial derivatives (joint delay) ---
    E, S = build_delay_prediction_matrices(N=N, dt=dt, delay_samples=2)

    def joint_block(j: int) -> Array:
        row = (j - 1) * 7
        return S[row:row + 6, 0:6]

    assert np.allclose(joint_block(1), 0), "q_phys_1 must have zero u_0 dependence"
    assert np.allclose(joint_block(2), 0), "q_phys_2 must have zero u_0 dependence"
    assert np.allclose(joint_block(3), dt * np.eye(6)), "q_phys_3 must have d/du_0 = dt*I"
    print("[mpc_delay_aware.prediction self-test] 1/4 PASS: "
          "u_0 first affects q_phys at stage 3 (dt*I), zero at stages 1-2")

    # --- 2: insertion undelayed ---
    def L_entry(j: int, r: int) -> float:
        row = (j - 1) * 7 + 6
        col = r * 7 + 6
        return float(S[row, col])

    assert abs(L_entry(1, 0) - dt) < 1e-15, "d(L_1)/du_L0 must equal dt (insertion undelayed)"
    print("[mpc_delay_aware.prediction self-test] 2/4 PASS: insertion is undelayed")

    # --- 3: condensed (E,S) vs an independent step-by-step recursion ---
    def simulate_recursion(z0_aug: Array, v: Array, N: int, dt: float, d: int = 2) -> Array:
        c_k, c_km1, Lk = z0_aug[:6], z0_aug[6:12], z0_aug[12]
        v = v.reshape(N, 7)
        c_hist = {0: c_k, -1: c_km1}
        c = c_k.copy()
        for i in range(1, N + 1):
            c = c + dt * v[i - 1, :6]
            c_hist[i] = c
        L = Lk
        out = np.zeros((N, 7))
        for j in range(1, N + 1):
            out[j - 1, :6] = c_hist[j - d]
            L = L + dt * v[j - 1, 6]
            out[j - 1, 6] = L
        return out.reshape(-1)

    z0_aug = rng.normal(size=13) * 0.01
    v = rng.normal(size=N * 7) * 0.05
    pred_matrix = predict_z_phys(E=E, S=S, z0_aug=z0_aug, v=v, N=N).reshape(-1)
    pred_recursion = simulate_recursion(z0_aug, v, N, dt, d=2)
    assert np.allclose(pred_matrix, pred_recursion, atol=1e-12), (
        f"condensed prediction != independent recursion, "
        f"max diff = {np.max(np.abs(pred_matrix - pred_recursion))}"
    )
    eps = 1.0e-6
    for col in (0, 6, 13, 20, 35, 41):
        v_plus = v.copy()
        v_plus[col] += eps
        fd = (
            predict_z_phys(E=E, S=S, z0_aug=z0_aug, v=v_plus, N=N).reshape(-1)
            - predict_z_phys(E=E, S=S, z0_aug=z0_aug, v=v, N=N).reshape(-1)
        ) / eps
        assert np.allclose(fd, S[:, col], atol=1.0e-9), f"finite-difference mismatch at column {col}"
    print("[mpc_delay_aware.prediction self-test] 3/4 PASS: matches an independent recursion "
          "exactly, and finite differences match S columns")

    # --- 4: delay_samples=0 reproduces the existing (undelayed) model ---
    E0, S0 = build_delay_prediction_matrices(N=N, dt=dt, delay_samples=0)
    E_std, S_std = build_standard_prediction_matrices(N=N, dt=dt, n=7)
    assert np.allclose(S0, S_std), "S mismatch at delay_samples=0"
    state7 = rng.normal(size=7) * 0.02
    z0_aug_equiv = np.concatenate([state7[:6], np.zeros(6), [state7[6]]])
    pred_delay0 = (E0 @ z0_aug_equiv).reshape(N, 7)
    pred_std = (E_std @ state7).reshape(N, 7)
    assert np.allclose(pred_delay0, pred_std), "E mismatch at delay_samples=0"
    print("[mpc_delay_aware.prediction self-test] 4/4 PASS: delay_samples=0 "
          "reproduces the existing MPC's E/S exactly")

    print("[mpc_delay_aware.prediction self-test] ALL PASS")


def build_standard_prediction_matrices(*, N: int, dt: float, n: int = 7) -> tuple[Array, Array]:
    """The EXISTING (undelayed) E/S from ConfigurationTrackingMPC, reproduced
    here only for the delay_samples=0-equivalent regression check -- q_phys_j
    = z0 + dt*sum(u_0..u_{j-1}) for every stage, no shift."""
    identity_state = np.eye(n, dtype=float)
    E = np.kron(np.ones((N, 1)), identity_state)
    S = np.kron(np.tril(np.ones((N, N), dtype=float)), dt * identity_state)
    return E, S


if __name__ == "__main__":
    run_self_test()
