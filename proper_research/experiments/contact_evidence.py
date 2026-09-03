"""Measurements that separate "contact is a disturbance" from "contact is model".

The claim worth establishing for neurovascular anatomy is not that contact makes
tracking harder — everyone accepts that — but the sharper one:

    contact is not an additive disturbance on the output; it changes the
    input-output sensitivity itself, so no disturbance observer can substitute
    for putting it in the model.

That claim is falsifiable, and this module measures the things that would
falsify it.

Why the distinction has teeth
-----------------------------
Write the plant output as ``p = f(z)``.  A controller linearises it as
``p ~ p_nom + J (z - z_ref) + d``.

* If contact were a **disturbance**, it would enter only through ``d``: an
  additive, input-independent offset.  ``J`` would be unchanged, ``d`` would be
  a function of where you are along the path but not of how you got there, and
  a disturbance estimator — which the beam-output MPC already runs — would
  absorb it.  Under this hypothesis the contact-free controller should match
  the contact-aware one once its estimator has converged.

* If contact is **structural**, it changes ``J``.  Then ``d_hat`` is not a
  disturbance at all: it is ``(J_contact - J_free)(z - z_ref)``, a quantity
  proportional to the controller's own tracking error.  An estimator chasing it
  is chasing a moving target that its own actions create, and no filter setting
  fixes that.

The second hypothesis makes three predictions this module tests, in increasing
order of cost:

1. **Offline** — the two Jacobians differ, and not just in magnitude: their row
   spaces rotate.  A disturbance cannot rotate a sensitivity.
   (:func:`jacobian_divergence`)

2. **From one closed-loop run** — ``d_hat`` regressed on the configuration
   tracking error recovers the Jacobian gap.  If that regression has high R^2
   and its coefficient matrix matches ``J_contact - J_free``, then what the
   observer is calling a disturbance is literally the missing model term.
   (:func:`disturbance_or_model`)

3. **Across anatomies** — the penalty for using the contact-free model grows
   with contact engagement.  A disturbance-hypothesis penalty would be flat in
   engagement once the observer converges.
   (:func:`engagement_dose_response`, fed by the study runner)

The first is cheap and is the figure to draw first: it needs no controller and
no simulation, only the two models and a set of states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

Array = np.ndarray


# --------------------------------------------------------------------------
# contact engagement: the dose
# --------------------------------------------------------------------------
def contact_engagement(model: Any) -> dict[str, float]:
    """Read how hard the beam is pressing, from the model's own diagnostics.

    Uses the same fields the global planner sniffs — ``gap_min``, ``W_cf``,
    ``F_nodes``, ``contact_weights`` — so the engagement number in a study is
    the same quantity the planner reports, not a second definition of contact.

    Deliberately a *measured* dose rather than the lumen radius: radius is a
    knob, engagement is what the beam actually experiences, and only the latter
    is comparable across anatomies with different bend patterns.
    """
    out = {
        "gap_min_m": np.nan,
        "contact_energy_J": np.nan,
        "contact_force_norm_N": np.nan,
        "contacting_node_fraction": np.nan,
        "contact_active": 0.0,
    }
    info: dict[str, Any] = {}
    for source in ("get_last_diag", "get_last_jacobian_diag"):
        method = getattr(model, source, None)
        if callable(method):
            try:
                value = method()
                if isinstance(value, dict):
                    info.update(value)
            except Exception:
                pass
    cache_info = getattr(getattr(model, "cache", None), "info", None)
    parts = (cache_info or {}).get("parts", {}) if isinstance(cache_info, dict) else {}
    if isinstance(parts, dict):
        info.update(parts)

    gap = info.get("gap_min")
    if gap is not None and np.isfinite(gap):
        out["gap_min_m"] = float(gap)
    energy = info.get("W_cf")
    if energy is not None and np.isfinite(energy):
        out["contact_energy_J"] = float(energy)
    forces = info.get("F_nodes")
    if forces is not None:
        forces = np.asarray(forces, dtype=float)
        if forces.size:
            out["contact_force_norm_N"] = float(np.linalg.norm(forces))
    weights = info.get("contact_weights")
    if weights is not None:
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if weights.size:
            out["contacting_node_fraction"] = float(np.mean(weights > 1e-9))
    out["contact_active"] = float(
        (np.isfinite(out["contact_energy_J"]) and out["contact_energy_J"] > 0.0)
        or (
            np.isfinite(out["contacting_node_fraction"])
            and out["contacting_node_fraction"] > 0.0
        )
    )
    return out


# --------------------------------------------------------------------------
# 1. offline: how the two Jacobians differ
# --------------------------------------------------------------------------
def _row_space_basis(jacobian: Array, tolerance: float = 1e-12) -> Array:
    """Orthonormal basis for the row space of a 3x7 Jacobian, as columns."""
    u, s, vt = np.linalg.svd(np.asarray(jacobian, dtype=float), full_matrices=False)
    keep = s > max(tolerance, s[0] * 1e-10) if s.size else np.zeros(0, dtype=bool)
    return vt[keep].T


def principal_angles_deg(first: Array, second: Array) -> Array:
    """Principal angles between the row spaces of two Jacobians, in degrees.

    This is the structural test.  A purely additive disturbance leaves the
    sensitivity untouched, so every angle would be zero.  A non-zero angle means
    the set of joint motions that move the tip is *different* under contact —
    the controller is not merely biased, it is pushing in directions that no
    longer do what it thinks.

    Returned in **ascending** order: element 0 is the smallest principal angle
    and element -1 the largest.  The ordering is not cosmetic.  ``np.linalg.svd``
    returns singular values descending, and the angles are their arccosines, so
    the first entry is the *most aligned* pair of directions — which is very
    often exactly zero, because two row spaces that share even one direction
    have a zero smallest angle.  The number that says "these subspaces are
    different" is the **last** one.
    """
    a = _row_space_basis(first)
    b = _row_space_basis(second)
    if a.size == 0 or b.size == 0:
        return np.full(3, np.nan)
    singular = np.linalg.svd(a.T @ b, compute_uv=False)
    angles = np.degrees(np.arccos(np.clip(singular, -1.0, 1.0)))
    angles = np.sort(angles)
    padded = np.full(3, np.nan)
    padded[: angles.size] = angles
    return padded


def directional_prediction_error(
    contact: Array, contact_free: Array, direction: Array
) -> float:
    """Relative one-step prediction error along one commanded direction.

    ``||(J_c - J_n) dz|| / ||J_c dz||`` — how wrong the contact-free model is
    about the tip motion the controller is actually asking for.  Averaged over
    a run's commands this is the number that predicts closed-loop degradation,
    and it can be far larger than the Frobenius gap when the command happens to
    lie in a direction contact affects.
    """
    direction = np.asarray(direction, dtype=float).reshape(7)
    scale = float(np.linalg.norm(np.asarray(contact, dtype=float) @ direction))
    if scale <= 1e-30:
        return float("nan")
    gap = np.asarray(contact, dtype=float) - np.asarray(contact_free, dtype=float)
    return float(np.linalg.norm(gap @ direction) / scale)


@dataclass
class JacobianDivergence:
    """Per-state comparison of the contact and contact-free Jacobians."""

    states: Array
    frobenius_relative: Array
    spectral_relative: Array
    principal_angles_deg: Array
    gain_ratio: Array
    condition_contact: Array
    condition_contact_free: Array
    per_actuator_relative: Array
    engagement: dict[str, Array]

    def summary(self) -> dict[str, Any]:
        def stats(values: Array) -> dict[str, float]:
            finite = np.asarray(values, dtype=float).reshape(-1)
            finite = finite[np.isfinite(finite)]
            if not finite.size:
                return {"max": float("nan"), "rms": float("nan"), "median": float("nan")}
            return {
                "max": float(np.max(finite)),
                "rms": float(np.sqrt(np.mean(finite**2))),
                "median": float(np.median(finite)),
            }

        return {
            "samples": int(self.states.shape[0]),
            "frobenius_relative": stats(self.frobenius_relative),
            "spectral_relative": stats(self.spectral_relative),
            # Ascending order, so [-1] is the largest angle: the one that says
            # the two row spaces are genuinely different subspaces. [0] is the
            # smallest and is near zero whenever they share any direction, which
            # they almost always do.
            "largest_principal_angle_deg": stats(self.principal_angles_deg[:, -1]),
            "smallest_principal_angle_deg": stats(self.principal_angles_deg[:, 0]),
            "gain_ratio_smallest_singular_value": stats(self.gain_ratio[:, -1]),
            "condition_contact": stats(self.condition_contact),
            "condition_contact_free": stats(self.condition_contact_free),
            # A column that is identically zero in the contact Jacobian has no
            # defined *relative* gap, so it stays NaN rather than being reported
            # as a zero difference it did not earn.
            "per_actuator_relative_max": [
                float(np.max(column[np.isfinite(column)]))
                if np.any(np.isfinite(column)) else float("nan")
                for column in self.per_actuator_relative.T
            ],
            "engagement": {
                key: stats(value) for key, value in self.engagement.items()
            },
        }


def jacobian_divergence(
    *,
    contact_provider: Callable[[Array], Array],
    contact_free_provider: Callable[[Array], Array],
    states: Any,
    contact_model: Any | None = None,
) -> JacobianDivergence:
    """Compare the two Jacobians over a set of configurations.

    No controller, no closed loop, no plant stepping — just the two models
    evaluated at the same states.  Run this first: if the two Jacobians agree
    to a fraction of a percent everywhere, there is no structural contact effect
    on this anatomy and the rest of the study will find nothing.

    ``contact_model`` is optional; when given, the beam's own contact
    diagnostics are recorded alongside each comparison so divergence can be
    plotted against engagement rather than against sample index.
    """
    states = np.asarray(states, dtype=float).reshape(-1, 7)
    count = states.shape[0]
    frobenius = np.empty(count)
    spectral = np.empty(count)
    angles = np.empty((count, 3))
    gains = np.empty((count, 3))
    condition_c = np.empty(count)
    condition_n = np.empty(count)
    per_actuator = np.empty((count, 7))
    engagement: dict[str, list[float]] = {}

    for index, state in enumerate(states):
        jc = np.asarray(contact_provider(state), dtype=float).reshape(3, 7)
        jn = np.asarray(contact_free_provider(state), dtype=float).reshape(3, 7)
        gap = jc - jn
        norm_c = float(np.linalg.norm(jc))
        frobenius[index] = float(np.linalg.norm(gap)) / max(norm_c, 1e-30)
        spectral[index] = float(np.linalg.norm(gap, 2)) / max(
            float(np.linalg.norm(jc, 2)), 1e-30
        )
        angles[index] = principal_angles_deg(jc, jn)
        sc = np.linalg.svd(jc, compute_uv=False)
        sn = np.linalg.svd(jn, compute_uv=False)
        gains[index] = sc / np.maximum(sn, 1e-30)
        condition_c[index] = float(sc[0] / max(sc[-1], 1e-30))
        condition_n[index] = float(sn[0] / max(sn[-1], 1e-30))
        for column in range(7):
            reference = float(np.linalg.norm(jc[:, column]))
            per_actuator[index, column] = (
                float(np.linalg.norm(gap[:, column])) / reference
                if reference > 1e-30
                else np.nan
            )
        if contact_model is not None:
            for key, value in contact_engagement(contact_model).items():
                engagement.setdefault(key, []).append(value)

    return JacobianDivergence(
        states=states,
        frobenius_relative=frobenius,
        spectral_relative=spectral,
        principal_angles_deg=angles,
        gain_ratio=gains,
        condition_contact=condition_c,
        condition_contact_free=condition_n,
        per_actuator_relative=per_actuator,
        engagement={k: np.asarray(v, dtype=float) for k, v in engagement.items()},
    )


# --------------------------------------------------------------------------
# 2. from one run: is the "disturbance" actually the missing model term?
# --------------------------------------------------------------------------
@dataclass
class DisturbanceVerdict:
    """Whether ``d_hat`` behaves like a disturbance or like model error."""

    samples: int
    r_squared_total: float
    r_squared_per_axis: list[float]
    fitted_matrix: Array
    jacobian_gap: Array | None
    gap_recovery_relative_error: float
    residual_rms_m: float
    disturbance_rms_m: float
    autocorrelation_lag1: float
    verdict: str
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "r_squared_total": self.r_squared_total,
            "r_squared_per_axis": self.r_squared_per_axis,
            "fitted_matrix": self.fitted_matrix.tolist(),
            "jacobian_gap": (
                None if self.jacobian_gap is None else self.jacobian_gap.tolist()
            ),
            "gap_recovery_relative_error": self.gap_recovery_relative_error,
            "residual_rms_m": self.residual_rms_m,
            "disturbance_rms_m": self.disturbance_rms_m,
            "autocorrelation_lag1": self.autocorrelation_lag1,
            "verdict": self.verdict,
            "detail": self.detail,
        }


def disturbance_or_model(
    *,
    estimated_residual: Any,
    state: Any,
    reference_state: Any,
    jacobian_gap: Any | None = None,
    r_squared_threshold: float = 0.5,
    recovery_threshold: float = 0.5,
) -> DisturbanceVerdict:
    """The decisive single-run test.

    The beam-output MPC estimates ``d_hat = p_meas - (p_nom + J_n (z - z_ref))``.
    If the plant is really ``p_nom + J_c (z - z_ref)`` then, algebraically,

        d_hat = (J_c - J_n) (z - z_ref)

    so ``d_hat`` is not an exogenous signal at all — it is a **linear function of
    the controller's own configuration tracking error**, with the Jacobian gap
    as its coefficient.

    This function regresses ``d_hat`` on ``(z - z_ref)`` and reports two things:

    * **R^2** — how much of the so-called disturbance is explained by the
      controller's own tracking error.  High R^2 means the estimator is chasing
      its own tail.
    * **gap recovery** — how close the fitted 3x7 coefficient matrix is to the
      Jacobian gap you can compute independently.  If it matches, the
      identification is unambiguous: the disturbance *is* the missing model
      term, not something the model happens to correlate with.

    A genuine disturbance would give low R^2 and a fitted matrix that matched
    nothing.

    Inputs are the columns your MPC already logs: ``beam_residual_*_m`` for
    ``estimated_residual``, ``q*_actual`` / ``insertion_actual`` for ``state``,
    ``q*_reference`` / ``insertion_reference`` for ``reference_state``.
    """
    residual = np.asarray(estimated_residual, dtype=float).reshape(-1, 3)
    state = np.asarray(state, dtype=float).reshape(-1, 7)
    reference_state = np.asarray(reference_state, dtype=float).reshape(-1, 7)
    count = min(residual.shape[0], state.shape[0], reference_state.shape[0])
    residual, state, reference_state = (
        residual[:count],
        state[:count],
        reference_state[:count],
    )
    finite = np.all(np.isfinite(residual), axis=1) & np.all(
        np.isfinite(state - reference_state), axis=1
    )
    residual = residual[finite]
    error = (state - reference_state)[finite]
    if residual.shape[0] < 10:
        raise ValueError(
            "Need at least ten finite samples to regress the residual; the run "
            "may not have logged beam_residual_*_m."
        )

    # Least squares with an intercept: a real constant offset should be allowed
    # to absorb into the intercept rather than inflating R^2.
    design = np.hstack([error, np.ones((error.shape[0], 1))])
    solution, *_ = np.linalg.lstsq(design, residual, rcond=None)
    fitted = design @ solution
    residual_after = residual - fitted
    total_variance = float(np.sum((residual - residual.mean(axis=0)) ** 2))
    r_squared_total = (
        1.0 - float(np.sum(residual_after**2)) / total_variance
        if total_variance > 1e-30
        else float("nan")
    )
    per_axis = []
    for axis in range(3):
        variance = float(np.sum((residual[:, axis] - residual[:, axis].mean()) ** 2))
        per_axis.append(
            1.0 - float(np.sum(residual_after[:, axis] ** 2)) / variance
            if variance > 1e-30
            else float("nan")
        )
    matrix = solution[:7].T  # 3x7

    recovery = float("nan")
    gap = None
    if jacobian_gap is not None:
        gap = np.asarray(jacobian_gap, dtype=float).reshape(3, 7)
        scale = float(np.linalg.norm(gap))
        if scale > 1e-30:
            recovery = float(np.linalg.norm(matrix - gap) / scale)

    centred = residual - residual.mean(axis=0)
    denominator = float(np.sum(centred[:-1] * centred[:-1]))
    autocorrelation = (
        float(np.sum(centred[:-1] * centred[1:]) / denominator)
        if denominator > 1e-30
        else float("nan")
    )

    explained = np.isfinite(r_squared_total) and r_squared_total >= r_squared_threshold
    recovered = np.isfinite(recovery) and recovery <= recovery_threshold
    if explained and recovered:
        verdict = "structural"
        detail = (
            f"{100 * r_squared_total:.0f}% of the estimated disturbance is a "
            "linear function of the controller's own configuration tracking "
            f"error, and the fitted coefficient matrix reproduces the Jacobian "
            f"gap to {100 * recovery:.0f}% relative error. This is not a "
            "disturbance; it is the missing model term, and no observer setting "
            "removes it."
        )
    elif explained:
        verdict = "structural (unconfirmed)"
        detail = (
            f"{100 * r_squared_total:.0f}% of the estimated disturbance is "
            "explained by the configuration tracking error, so it is "
            "input-dependent rather than exogenous. Supply jacobian_gap to "
            "confirm it is the contact term specifically."
        )
    else:
        verdict = "disturbance-like"
        detail = (
            f"Only {100 * max(r_squared_total, 0.0):.0f}% of the residual tracks "
            "the configuration error, so on this run it behaves like an "
            "exogenous offset an observer can absorb."
        )

    return DisturbanceVerdict(
        samples=int(residual.shape[0]),
        r_squared_total=float(r_squared_total),
        r_squared_per_axis=[float(v) for v in per_axis],
        fitted_matrix=matrix,
        jacobian_gap=gap,
        gap_recovery_relative_error=recovery,
        residual_rms_m=float(np.sqrt(np.mean(residual_after**2))),
        disturbance_rms_m=float(np.sqrt(np.mean(residual**2))),
        autocorrelation_lag1=autocorrelation,
        verdict=verdict,
        detail=detail,
    )


# --------------------------------------------------------------------------
# 3. across anatomies: does the penalty grow with engagement?
# --------------------------------------------------------------------------
def engagement_dose_response(
    *,
    engagement: Sequence[float],
    contact_free_metric: Sequence[float],
    contact_aware_metric: Sequence[float],
    metric_name: str = "tip rms error",
) -> dict[str, Any]:
    """Fit the penalty for using the contact-free model against engagement.

    The disturbance hypothesis predicts a **flat** relationship: once the
    observer has converged, more contact means a larger offset to absorb but not
    a worse controller.  The structural hypothesis predicts a **rising** one,
    because the sensitivity error grows with engagement and the observer cannot
    compensate a gain error.

    Reports the slope with a bootstrap interval, so "it rises" is a measurement
    rather than an impression.  With three or four anatomies the interval will
    be wide; report it honestly rather than quoting a bare slope.
    """
    engagement = np.asarray(engagement, dtype=float).reshape(-1)
    free = np.asarray(contact_free_metric, dtype=float).reshape(-1)
    aware = np.asarray(contact_aware_metric, dtype=float).reshape(-1)
    if not (engagement.size == free.size == aware.size):
        raise ValueError("engagement and the two metric series must be the same length.")
    finite = np.isfinite(engagement) & np.isfinite(free) & np.isfinite(aware)
    engagement, free, aware = engagement[finite], free[finite], aware[finite]
    if engagement.size < 3:
        raise ValueError("Need at least three anatomies to fit a dose response.")

    penalty = free - aware
    ratio = free / np.maximum(aware, 1e-30)
    design = np.vstack([engagement, np.ones_like(engagement)]).T
    coefficients, *_ = np.linalg.lstsq(design, penalty, rcond=None)
    slope, intercept = float(coefficients[0]), float(coefficients[1])
    predicted = design @ coefficients
    variance = float(np.sum((penalty - penalty.mean()) ** 2))
    r_squared = (
        1.0 - float(np.sum((penalty - predicted) ** 2)) / variance
        if variance > 1e-30
        else float("nan")
    )

    rng = np.random.default_rng(11)
    slopes = []
    for _ in range(2000):
        pick = rng.integers(0, engagement.size, engagement.size)
        try:
            fit, *_ = np.linalg.lstsq(design[pick], penalty[pick], rcond=None)
            slopes.append(float(fit[0]))
        except np.linalg.LinAlgError:
            continue
    interval = (
        [float(np.percentile(slopes, 5)), float(np.percentile(slopes, 95))]
        if slopes
        else [float("nan"), float("nan")]
    )

    rises = np.isfinite(interval[0]) and interval[0] > 0.0
    return {
        "metric": metric_name,
        "anatomies": int(engagement.size),
        "engagement": engagement.tolist(),
        "contact_free": free.tolist(),
        "contact_aware": aware.tolist(),
        "penalty": penalty.tolist(),
        "penalty_ratio": ratio.tolist(),
        "slope": slope,
        "intercept": intercept,
        "slope_90_percent_interval": interval,
        "r_squared": r_squared,
        "penalty_rises_with_engagement": bool(rises),
        "verdict": (
            "The cost of the contact-free model grows with contact engagement. "
            "A disturbance an observer can absorb would not do that."
            if rises
            else "No detectable rise. On this set of anatomies the contact-free "
            "model is not measurably penalised more as contact increases — "
            "either engagement did not vary enough, or the observer is coping."
        ),
    }


__all__ = [
    "DisturbanceVerdict",
    "JacobianDivergence",
    "contact_engagement",
    "directional_prediction_error",
    "disturbance_or_model",
    "engagement_dose_response",
    "jacobian_divergence",
    "principal_angles_deg",
]
