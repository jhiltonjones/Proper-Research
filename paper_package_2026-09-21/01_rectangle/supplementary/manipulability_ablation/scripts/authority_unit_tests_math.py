"""Regression tests C and D (spec section 13) -- pure math on
authority_objective.py, no QP/controller integration needed yet."""
import sys
sys.path.insert(0, "/home/jack/.claude/jobs/3710eca5/tmp")
import numpy as np
from authority_objective import phi_authority, logdet_authority, build_Jbar, nullspace_projector

rng = np.random.default_rng(1)

# ======================================================================
# Test D: SVD-based phi equals direct logdet(I + Jbar Jbar^T)
# ======================================================================
print("=== Test D: SVD/logdet equivalence ===")
max_err = 0.0
for trial in range(20):
    Jbar = rng.normal(size=(2, 7)) * rng.uniform(0.1, 5.0)
    phi_svd, sv = phi_authority(Jbar)
    phi_direct = logdet_authority(Jbar)
    err = abs(phi_svd - phi_direct)
    max_err = max(max_err, err)
print(f"max |phi_svd - phi_logdet| over 20 random trials = {max_err:.3e}")
assert max_err < 1e-9, "FAIL: SVD-based phi disagrees with direct logdet"
print("Test D PASS\n")

# ======================================================================
# Test C: finite near/at singularity -- no inf/nan, no barrier blowup
# ======================================================================
print("=== Test C: finite near/at singularity ===")
cases = []
# Jbar with one singular value exactly 0 (rank-deficient by construction)
J_rank1 = np.outer([1.0, 0.0], rng.normal(size=7))  # rank-1, one row identically 0-correlated
cases.append(("rank-deficient (2x7, effective rank 1)", J_rank1))
J_zero = np.zeros((2, 7))
cases.append(("all-zero Jbar (both singular values 0)", J_zero))
J_huge = rng.normal(size=(2, 7)) * 1.0e6
cases.append(("huge singular values (1e6 scale)", J_huge))
J_tiny = rng.normal(size=(2, 7)) * 1.0e-9
cases.append(("tiny singular values (1e-9 scale)", J_tiny))

for name, Jbar in cases:
    phi, sv = phi_authority(Jbar)
    finite = np.isfinite(phi) and np.all(np.isfinite(sv))
    print(f"  {name}: phi={phi:.6e}  sv={np.round(sv,6).tolist()}  finite={finite}")
    assert finite, f"FAIL: non-finite result for {name}"
    # explicitly must NOT blow up like a barrier (1/sigma) would near sv=0
    if "tiny" in name or "all-zero" in name or "rank-deficient" in name:
        assert phi < 1.0, f"FAIL: phi={phi} looks like a barrier blew up near a small/zero singular value"
print("Test C PASS (all finite, no barrier-like blowup near small/zero singular values)\n")

# ======================================================================
# Extra: derivative-vanishing-at-zero property claimed in the spec
# d/dsigma log(1+sigma^2) = 2*sigma/(1+sigma^2) -> 0 as sigma -> 0
# ======================================================================
print("=== Extra check: d/dsigma log(1+sigma^2) -> 0 as sigma -> 0 (spec property 2) ===")
for sigma in (1e-6, 1e-3, 1e-1, 1.0, 10.0):
    deriv = 2 * sigma / (1 + sigma ** 2)
    print(f"  sigma={sigma:<8g} d(phi)/d(sigma) = {deriv:.6e}")
print("(confirms: near-zero singular directions get a near-zero gradient contribution -- by design, not a bug)\n")

# ======================================================================
# Nullspace projector sanity: idempotent, rank matches, orthogonal to J_scaled
# ======================================================================
print("=== Nullspace projector sanity (used later, but cheap to check now) ===")
S_Z = np.array([np.radians(0.5)] * 6 + [0.25e-3])
for trial in range(5):
    J_task = rng.normal(size=(2, 7))
    Pr, P_N, rank = nullspace_projector(J_task, S_Z)
    idempotent_err = np.max(np.abs(P_N @ P_N - P_N))
    symmetric_err = np.max(np.abs(P_N - P_N.T))
    J_s = J_task * S_Z[None, :]
    annihilation_err = np.max(np.abs(J_s @ P_N))
    print(f"  trial={trial}: rank={rank}  idempotent_err={idempotent_err:.2e}  "
          f"symmetric_err={symmetric_err:.2e}  ||J_s @ P_N||_max={annihilation_err:.2e}")
    assert idempotent_err < 1e-9 and symmetric_err < 1e-9 and annihilation_err < 1e-9
print("Nullspace projector sanity PASS\n")

print("ALL MATH-ONLY REGRESSION TESTS PASSED")
