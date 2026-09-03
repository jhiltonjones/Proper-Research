"""Self-test for contact_evidence.py — synthetic only, no project imports.

Every check here is a *falsification* check: each one constructs data under a
known hypothesis and asserts the module reaches the verdict that hypothesis
deserves.  A test that only ran the functions and checked they returned floats
would pass on a module that always answered "structural", which is the answer
we most need to be able to disbelieve.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
# Layout-tolerant: works whether this file sits beside the modules it tests
# (experiments/) or in a separate tests/ directory beside them.
for _root in (_HERE, _HERE.parent, _HERE.parent.parent):
    for _sub in ("", "experiments", "analysis", "controllers"):
        _path = _root / _sub if _sub else _root
        if _path.is_dir() and str(_path) not in sys.path:
            sys.path.insert(0, str(_path))

import contact_evidence as E  # noqa: E402

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((PASS if condition else FAIL, name, detail))


rng = np.random.default_rng(7)


# --------------------------------------------------------------------------
# 1. principal angles: identical -> 0, rotated -> nonzero, scaled -> still 0
# --------------------------------------------------------------------------
J = rng.normal(size=(3, 7))
check("identical Jacobians give zero principal angles",
      float(np.max(E.principal_angles_deg(J, J))) < 1e-6,
      f"max {float(np.max(E.principal_angles_deg(J, J))):.2e} deg")

# A pure gain change is the disturbance-like case for the *structural* test:
# scaling every row leaves the row space untouched, so the angles must stay 0.
check("a pure gain change does not rotate the row space",
      float(np.max(E.principal_angles_deg(J, 3.7 * J))) < 1e-6,
      "scaling by 3.7")

rotated = J.copy()
rotated[:, 6] += 0.8 * J[:, 0]   # mixes insertion into a joint direction
angle = float(np.max(E.principal_angles_deg(J, rotated)))
check("a genuine sensitivity change does rotate the row space",
      angle > 1.0, f"largest principal angle {angle:.2f} deg")


# --------------------------------------------------------------------------
# 2. jacobian_divergence over a set of states
# --------------------------------------------------------------------------
base = rng.normal(size=(3, 7))
gap_matrix = 0.15 * rng.normal(size=(3, 7))


def contact_provider(state):
    return base + gap_matrix * (1.0 + 0.1 * float(state[0]))


def free_provider(state):
    return base


states = rng.normal(size=(25, 7))
divergence = E.jacobian_divergence(
    contact_provider=contact_provider,
    contact_free_provider=free_provider,
    states=states,
)
summary = divergence.summary()
check("divergence reports one row per state",
      divergence.frobenius_relative.shape == (25,), str(divergence.frobenius_relative.shape))
check("divergence detects a nonzero relative gap",
      summary["frobenius_relative"]["median"] > 0.01,
      f"median {summary['frobenius_relative']['median']:.4f}")
check("divergence summary is JSON-shaped",
      isinstance(summary["largest_principal_angle_deg"]["max"], float))

identical = E.jacobian_divergence(
    contact_provider=free_provider, contact_free_provider=free_provider, states=states
)
check("identical providers give a zero gap (no false positive)",
      float(np.max(identical.frobenius_relative)) < 1e-12,
      f"max {float(np.max(identical.frobenius_relative)):.2e}")


# --------------------------------------------------------------------------
# 3. disturbance_or_model — the decisive test, both ways round
# --------------------------------------------------------------------------
samples = 300
reference_state = rng.normal(size=(samples, 7))
tracking_error = 0.02 * rng.normal(size=(samples, 7))
state = reference_state + tracking_error

# (a) structural: d_hat IS the gap acting on the tracking error.
structural_d = tracking_error @ gap_matrix.T
verdict = E.disturbance_or_model(
    estimated_residual=structural_d,
    state=state,
    reference_state=reference_state,
    jacobian_gap=gap_matrix,
)
check("structural data is called structural", verdict.verdict.startswith("structural"),
      f"{verdict.verdict}, R^2={verdict.r_squared_total:.4f}")
check("the fitted matrix recovers the Jacobian gap",
      verdict.gap_recovery_relative_error < 1e-6,
      f"recovery error {verdict.gap_recovery_relative_error:.2e}")

# (b) disturbance: an offset that drifts with time and ignores the input.
drift = np.linspace(0.0, 1.0, samples)[:, None] * np.array([[1e-4, -2e-4, 5e-5]])
noise = 1e-6 * rng.normal(size=(samples, 3))
verdict_d = E.disturbance_or_model(
    estimated_residual=drift + noise,
    state=state,
    reference_state=reference_state,
    jacobian_gap=gap_matrix,
)
check("an input-independent drift is NOT called structural",
      verdict_d.verdict == "disturbance-like",
      f"{verdict_d.verdict}, R^2={verdict_d.r_squared_total:.4f}")

# (c) structural but with no gap supplied: must not claim confirmation.
verdict_u = E.disturbance_or_model(
    estimated_residual=structural_d, state=state, reference_state=reference_state
)
check("without a supplied gap the verdict is marked unconfirmed",
      "unconfirmed" in verdict_u.verdict, verdict_u.verdict)

# (d) a gap that is real but the *wrong* one: high R^2, poor recovery.
verdict_w = E.disturbance_or_model(
    estimated_residual=structural_d,
    state=state,
    reference_state=reference_state,
    jacobian_gap=rng.normal(size=(3, 7)),
)
check("a wrong gap fails the recovery check even at high R^2",
      verdict_w.gap_recovery_relative_error > 0.5,
      f"R^2={verdict_w.r_squared_total:.3f}, recovery={verdict_w.gap_recovery_relative_error:.3f}")


# --------------------------------------------------------------------------
# 4. dose response — rising and flat
# --------------------------------------------------------------------------
engagement = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.4])
aware = np.array([0.30, 0.31, 0.32, 0.33, 0.34, 0.35])
rising = E.engagement_dose_response(
    engagement=engagement,
    contact_free_metric=aware + 0.9 * engagement + 0.01,
    contact_aware_metric=aware,
)
check("a rising penalty is detected", rising["penalty_rises_with_engagement"],
      f"slope {rising['slope']:.3f}, 90% {rising['slope_90_percent_interval']}")

flat = E.engagement_dose_response(
    engagement=engagement,
    contact_free_metric=aware + 0.05,
    contact_aware_metric=aware,
)
check("a flat penalty is NOT called rising", not flat["penalty_rises_with_engagement"],
      f"slope {flat['slope']:.3e}, 90% {flat['slope_90_percent_interval']}")

try:
    E.engagement_dose_response(
        engagement=[0.0, 1.0], contact_free_metric=[1.0, 2.0], contact_aware_metric=[1.0, 1.0]
    )
    check("two anatomies are refused", False, "no error raised")
except ValueError:
    check("two anatomies are refused", True, "ValueError as expected")


# --------------------------------------------------------------------------
# 5. directional prediction error and engagement reader
# --------------------------------------------------------------------------
direction = np.zeros(7)
direction[0] = 1.0
value = E.directional_prediction_error(base + gap_matrix, base, direction)
expected = float(np.linalg.norm(gap_matrix[:, 0]) / np.linalg.norm((base + gap_matrix)[:, 0]))
check("directional error matches the closed form", abs(value - expected) < 1e-12,
      f"{value:.6f} vs {expected:.6f}")


class _Model:
    def get_last_diag(self):
        return {"gap_min": -1.2e-4, "W_cf": 3.4e-6,
                "F_nodes": np.array([[0.0, 0.0, 0.01]]),
                "contact_weights": np.array([0.0, 1.0, 1.0, 0.0])}


reading = E.contact_engagement(_Model())
check("engagement reads the model's own diagnostics",
      abs(reading["contacting_node_fraction"] - 0.5) < 1e-12
      and reading["contact_active"] == 1.0,
      f"fraction {reading['contacting_node_fraction']}, active {reading['contact_active']}")


class _Silent:
    pass


quiet = E.contact_engagement(_Silent())
check("a model with no diagnostics degrades to NaN, not a crash",
      np.isnan(quiet["gap_min_m"]) and quiet["contact_active"] == 0.0)

# --------------------------------------------------------------------------
print("\ncontact_evidence self-test")
print("=" * 74)
for status, name, detail in results:
    print(f"  [{status}] {name}" + (f"   ({detail})" if detail else ""))
failures = sum(1 for status, _, _ in results if status == FAIL)
print("=" * 74)
print(f"  {len(results) - failures}/{len(results)} checks passed")
sys.exit(1 if failures else 0)
