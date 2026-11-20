"""Offline, hardware-free unit tests for ScalarJacobianAdaptation. Updated
2026-11-19 for the corrected (first-differenced) estimator -- see
jacobian_adaptation.py's module docstring. The first sample ever is now
unconditionally accepted (nothing to difference against yet), so several
fixtures below now include an explicit warm-up call before the behaviour
under test; this is a FIXTURE update to match the corrected design, not a
loosened assertion -- every original test's pass/fail condition is
preserved, including `test_outlier_residual_rejected` (preserved exactly
per the 2026-11-19 review's explicit instruction: outlier rejection
happens before any differencing logic, so only the removed
`prev_bias_estimate` kwarg changed).

Run: python3 -m proper_research.controllers.mpc_delay_aware.test_jacobian_adaptation
"""
from __future__ import annotations

import numpy as np

from proper_research.controllers.mpc_delay_aware.jacobian_adaptation import (
    JacobianAdaptationConfig,
    ScalarJacobianAdaptation,
)

RNG = np.random.default_rng(0)


def _make_J() -> np.ndarray:
    # A plausible, well-conditioned (3,6) beam-tip Jacobian.
    J = np.array([
        [0.8, -0.1, 0.05, 0.0, 0.02, -0.01],
        [0.1, 0.9, -0.05, 0.03, 0.0, 0.02],
        [0.02, -0.03, 0.95, -0.01, 0.01, 0.0],
    ])
    return J


def test_zero_at_coincidence():
    """dq=0 provides no information about g (u=0 identically). Needs a
    warm-up sample first: the FIRST sample ever is unconditionally
    accepted (nothing to difference against), so "zero excitation" must
    be demonstrated as the SECOND call, where du(k)=u(k)-u(k-1)=0 because
    BOTH samples sit at dq=0."""
    cfg = JacobianAdaptationConfig(mode="active")
    est = ScalarJacobianAdaptation(cfg)
    J = _make_J()
    dq0 = np.zeros(6)
    warm = est.step(
        instantaneous_residual=np.zeros(3), delta_q6=dq0, j_nominal6=J, frame_index=0, sample_age_s=0.01,
    )
    assert warm["accepted"] and warm["first_sample"]
    diag = est.step(
        instantaneous_residual=np.array([0.01, -0.005, 0.002]),
        delta_q6=dq0, j_nominal6=J, frame_index=1, sample_age_s=0.01,
    )
    assert not diag["accepted"] and "insufficient_excitation" in diag["reject_reason"], diag
    print("PASS: zero excitation (du=0 between two dq=0 samples) is rejected, not used to move g")


def test_recovers_known_scalar_correction():
    """Synthetic mismatch: true Jacobian = (1+g_true)*J[:, :6]. Feed
    noiseless samples with varying dq AND a constant bias (to also
    confirm the bias does not need to be separately subtracted -- the
    differencing cancels it automatically), verify g_hat converges to
    g_true within tolerance."""
    g_true = 0.18
    J = _make_J()
    cfg = JacobianAdaptationConfig(mode="active", excitation_floor_m=1.0e-5, forgetting_lambda=1.0)
    est = ScalarJacobianAdaptation(cfg)
    bias = np.array([0.0005, -0.0003, 0.0002])  # constant bias; differencing must cancel it, no separate input needed
    for k in range(200):
        dq = RNG.normal(scale=0.05, size=6)
        true_residual = g_true * (J @ dq) + bias
        diag = est.step(
            instantaneous_residual=true_residual,
            delta_q6=dq, j_nominal6=J, frame_index=k, sample_age_s=0.01,
        )
    assert diag["accepted"]
    err = abs(diag["g_hat"] - g_true)
    assert err < 0.01, f"g_hat={diag['g_hat']:.4f} vs g_true={g_true:.4f}, err={err:.4f}"
    print(f"PASS: recovered g_hat={diag['g_hat']:.4f} (true={g_true}) with a constant bias present, "
          f"{diag['n_accepted']} accepted")


def test_shadow_mode_never_affects_g_used():
    cfg = JacobianAdaptationConfig(mode="shadow", excitation_floor_m=1.0e-5, forgetting_lambda=1.0)
    est = ScalarJacobianAdaptation(cfg)
    J = _make_J()
    g_true = 0.3
    for k in range(50):
        dq = RNG.normal(scale=0.05, size=6)
        residual = g_true * (J @ dq)
        diag = est.step(
            instantaneous_residual=residual,
            delta_q6=dq, j_nominal6=J, frame_index=k, sample_age_s=0.01,
        )
        assert diag["g_used"] == 0.0, "shadow mode must never set g_used != 0"
    assert abs(diag["g_hat"] - g_true) < 0.03, "shadow mode should still have LEARNED internally"
    print(f"PASS: shadow mode learned g_hat={diag['g_hat']:.4f} but g_used stayed 0.0 throughout")


def test_bounds_respected():
    # max_plausible_residual_m raised here deliberately: this test isolates
    # the g-bound clip from the separate outlier-rejection gate (covered by
    # test_outlier_residual_rejected) -- a true g=5.0 mismatch legitimately
    # produces large residuals that the outlier gate would otherwise (and
    # correctly, for a real run) reject before the regression ever saw them.
    cfg = JacobianAdaptationConfig(
        mode="active", g_min=-0.3, g_max=0.5, excitation_floor_m=1.0e-5,
        forgetting_lambda=1.0, max_plausible_residual_m=10.0,
    )
    est = ScalarJacobianAdaptation(cfg)
    J = _make_J()
    g_true = 5.0  # way outside bounds
    for k in range(100):
        dq = RNG.normal(scale=0.05, size=6)
        residual = g_true * (J @ dq)
        diag = est.step(
            instantaneous_residual=residual,
            delta_q6=dq, j_nominal6=J, frame_index=k, sample_age_s=0.01,
        )
    assert diag["g_hat"] == cfg.g_max, diag
    print(f"PASS: g_hat clipped to g_max={cfg.g_max} despite true scale {g_true}")


def test_insufficient_excitation_rejected():
    """Needs a warm-up sample first (see test_zero_at_coincidence's
    docstring): tiny motion alone is not rejectable on the very first
    sample (nothing to compare to yet); it IS rejectable once there is a
    prior sample whose u is equally tiny (du ~= 0)."""
    cfg = JacobianAdaptationConfig(mode="active", excitation_floor_m=1.0e-3)
    est = ScalarJacobianAdaptation(cfg)
    J = _make_J()
    dq = np.full(6, 1.0e-6)  # tiny motion
    warm = est.step(instantaneous_residual=np.zeros(3), delta_q6=dq, j_nominal6=J, frame_index=0, sample_age_s=0.01)
    assert warm["accepted"] and warm["first_sample"]
    diag = est.step(
        instantaneous_residual=np.zeros(3),
        delta_q6=dq, j_nominal6=J, frame_index=1, sample_age_s=0.01,
    )
    assert not diag["accepted"] and "insufficient_excitation" in diag["reject_reason"]
    print("PASS: tiny motion (du~=0 vs the prior sample) correctly rejected as insufficient excitation")


def test_stale_sample_rejected():
    cfg = JacobianAdaptationConfig(mode="active", max_sample_age_s=0.25)
    est = ScalarJacobianAdaptation(cfg)
    J = _make_J()
    dq = RNG.normal(scale=0.05, size=6)
    diag = est.step(
        instantaneous_residual=np.zeros(3),
        delta_q6=dq, j_nominal6=J, frame_index=1, sample_age_s=0.9,
    )
    assert not diag["accepted"] and "stale_sample" in diag["reject_reason"]
    print("PASS: stale sample correctly rejected")


def test_duplicate_frame_rejected():
    cfg = JacobianAdaptationConfig(mode="active")
    est = ScalarJacobianAdaptation(cfg)
    J = _make_J()
    dq = RNG.normal(scale=0.05, size=6)
    diag1 = est.step(
        instantaneous_residual=0.1 * (J @ dq),
        delta_q6=dq, j_nominal6=J, frame_index=7, sample_age_s=0.01,
    )
    diag2 = est.step(
        instantaneous_residual=0.1 * (J @ dq),
        delta_q6=dq, j_nominal6=J, frame_index=7, sample_age_s=0.01,
    )
    assert diag1["accepted"] and not diag2["accepted"] and diag2["reject_reason"] == "duplicate_frame_index"
    print("PASS: duplicate camera frame correctly rejected on second sighting")


def test_outlier_residual_rejected():
    """PRESERVED per the 2026-11-19 review's explicit instruction ("preserve
    the outlier-rejection test"). Outlier rejection happens before any
    differencing logic (it is a per-sample, not a per-pair, sanity check),
    so this fixture is UNCHANGED from the original version except for
    removing the now-nonexistent `prev_bias_estimate` kwarg."""
    cfg = JacobianAdaptationConfig(mode="active", max_plausible_residual_m=0.05)
    est = ScalarJacobianAdaptation(cfg)
    J = _make_J()
    dq = RNG.normal(scale=0.05, size=6)
    diag = est.step(
        instantaneous_residual=np.array([0.2, 0.0, 0.0]),
        delta_q6=dq, j_nominal6=J, frame_index=1, sample_age_s=0.01,
    )
    assert not diag["accepted"] and diag["reject_reason"] == "implausible_residual_outlier"
    print("PASS: 200mm outlier residual correctly rejected (preserved check)")


def test_nan_input_falls_back_not_raises():
    cfg = JacobianAdaptationConfig(mode="active")
    est = ScalarJacobianAdaptation(cfg)
    J = _make_J()
    diag = est.step(
        instantaneous_residual=np.array([np.nan, 0.0, 0.0]),
        delta_q6=np.zeros(6), j_nominal6=J, frame_index=1, sample_age_s=0.01,
    )
    assert not diag["accepted"] and diag["reject_reason"] == "non_finite_input"
    assert diag["g_used"] == 0.0
    print("PASS: NaN input falls back to g_used=0.0 without raising")


def test_reset_clears_state():
    cfg = JacobianAdaptationConfig(mode="active", excitation_floor_m=1.0e-5, forgetting_lambda=1.0)
    est = ScalarJacobianAdaptation(cfg)
    J = _make_J()
    for k in range(50):
        dq = RNG.normal(scale=0.05, size=6)
        est.step(
            instantaneous_residual=0.3 * (J @ dq),
            delta_q6=dq, j_nominal6=J, frame_index=k, sample_age_s=0.01,
        )
    assert est.n_accepted > 0 and est._g_hat != 0.0
    est.reset()
    assert est.n_accepted == 0 and est._g_hat == 0.0 and est._S == 0.0 and est._C == 0.0
    print("PASS: reset() fully clears accumulated estimator state")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = []
    for t in tests:
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            failures.append((t.__name__, repr(exc)))
            print(f"FAIL {t.__name__}: {exc!r}")
    print(f"\n{len(tests)-len(failures)}/{len(tests)} jacobian_adaptation unit tests passed.")
    if failures:
        raise SystemExit(1)
