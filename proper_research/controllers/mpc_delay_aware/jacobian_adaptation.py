"""Online joint-Jacobian adaptation (2026-11-18, corrected 2026-11-19 after
review): a bounded, reduced-order correction to the OFFLINE schedule's
beam-tip Jacobian, estimated online from the SAME output residual the
existing bias estimator already computes
(`BeamOutputTrackingMPC._estimate_output_residual`) -- this module does
not introduce a second output-bias estimator.

====================================================================
CORRECTED DERIVATION (2026-11-19) -- read this before trusting g_hat
====================================================================

The first version of this module claimed the regression target
`r_inst(k) - d_hat(k-1)` observes `g * J_nominal[idx] @ dq(k)` directly.
That claim is WRONG in general and was only accidentally close to right
in the original unit tests, which used i.i.d.-random, mutually
INDEPENDENT joint offsets each tick -- not a real robot's smoothly
autocorrelated trajectory. Here is the corrected derivation, done
properly.

**Model.** At reference index `idx = idx_meas(k) = clip(control_index, 0,
S-1)` (the SAME index `_estimate_output_residual` uses -- NOT the delayed
horizon's own stage-0 index, which is a different, non-coincident point;
see `delay_aware_mpc.py`'s prediction-equation docstring), assume the true
generative model is

    p_meas(k) = p_nom[idx] + (J_nom[idx] + dJ[idx]) @ (z_meas(k)-z_ref[idx])
                + b(k) + e(k)

with `dJ[idx] = g * [J_nom[idx][:, :6], 0]` (joint columns only, ONE
global scalar `g`, DIMENSIONLESS -- the reduced-order assumption), `b(k)`
an unknown, assumed-slowly-varying bias (meters, frame drift etc. --
exactly what the EXISTING bias estimator exists to track), and `e(k)`
i.i.d. zero-mean noise (meters), `Cov(e)=sigma^2 I`.

REGRESSOR NAME, stated explicitly to avoid the ambiguous bare name `u`
(already overloaded elsewhere in this project for the MPC decision
variable/control input, rad/s and m/s -- NOT the same quantity):
writing `dq(k) = q_meas(k)-q_ref[idx][:6]` (RADIANS, (6,)), define

    eta_pred_m(k) := J_nom[idx][:, :6] @ dq(k)          -- METERS, (3,)

"eta_pred_m": the NOMINAL (un-adapted, offline-schedule) model's own
predicted beam-tip DISPLACEMENT caused by the joint offset `dq(k)`, using
only the schedule's existing joint-sensitivity block. It is a PREDICTED
POSITION OFFSET, not a rate/velocity and not the control input -- the
code variable is named `eta_pred_m` throughout this module and its
diagnostics dict for exactly this reason. With this name, the
instantaneous residual `_estimate_output_residual` already computes is
EXACTLY

    r_inst(k) = g * eta_pred_m(k) + b(k) + e(k)                      (*)      [meters, (3,)]

This part is uncontroversial. The problem is eliminating the unknown
`b(k)` without a second, independent bias estimator.

**The fix: first-difference against the last ACCEPTED sample, not the
EMA state.** `b(k)` is unknown but assumed slowly varying tick to tick
(100ms here). Differencing (*) against the previous ACCEPTED sample k'
(not necessarily k-1 if samples in between were rejected -- see "GAPS
BETWEEN ACCEPTED SAMPLES" below) cancels it to first order:

    dr(k) := r_inst(k) - r_inst(k')
           = g*(eta_pred_m(k) - eta_pred_m(k')) + [b(k)-b(k')] + [e(k)-e(k')]
          ~= g * d_eta_pred_m(k)  +  d_noise(k)                      [meters, (3,)]

    where d_eta_pred_m(k) := eta_pred_m(k) - eta_pred_m(k')           [meters, (3,)]
          Var(d_noise) = 2*sigma^2   (differencing two independent noise draws)

This is now a CLEAN regression for `g` (dimensionless: `d_eta_pred_m`'s
units of meters cancel against `dr`'s units of meters), valid PROVIDED
`b` truly changes negligibly between the two samples differenced (tested
explicitly below, scenario D) and PROVIDED `g` really is the same global
scalar at `idx(k)` and `idx(k')` even if the schedule index itself
changed between them (tested explicitly, scenario C).

**GAPS BETWEEN ACCEPTED SAMPLES.** `k'` is whichever sample was last
ACCEPTED, not necessarily `k-1` -- ticks rejected in between (stale,
duplicate, insufficient excitation, outlier) are simply skipped, never
differenced against. The measurement equation above is UNCHANGED by how
large the gap `k-k'` is: it only requires `b(k)~=b(k')` across whatever
real time elapsed between the two ACCEPTED samples, not across every
intervening tick. A long gap is therefore a WEAKER form of the same
"slowly varying bias" assumption (over a longer interval), not a
different equation -- demonstrated directly in
`test_consistency_across_schedule_changes_and_gaps.py`, alongside the
already-covered schedule-index-changing case (scenario C).

**Why the FIRST version's code was actually computing this already, but
regressing against the WRONG quantity.** With `disturbance_filter_alpha
= 0.0` (the project-wide default -- confirmed: neither
`run_mpc_delay_aware_insertion_anchor.py` nor
`run_mpc_delay_aware_vessel.py` ever overrides it), the EMA recursion
`filtered(k)=alpha*filtered(k-1)+(1-alpha)*instantaneous(k)` reduces
EXACTLY to `filtered(k-1) = instantaneous(k-1) = r_inst(k-1)`. So
`r_inst(k) - d_hat(k-1)` WAS numerically identical to `dr(k)` above all
along (that part of the implementation was correct) -- the bug was fitting
`g` by regressing this correct LHS against `eta_pred_m(k)` (the first
version's actual RLS regressor, then named bare `u`) instead of
`d_eta_pred_m(k) = eta_pred_m(k) - eta_pred_m(k-1)`. On a real, smooth
robot trajectory `eta_pred_m(k) ~= eta_pred_m(k-1)` tick to tick (adjacent
configurations are close), so `d_eta_pred_m(k) ~= 0` almost always even
while `eta_pred_m(k)` itself stays persistently large -- regressing the
(correctly near-zero) LHS against the (persistently large) WRONG
regressor systematically biases `g_hat` toward 0, hiding a real mismatch.
This did not show up in the original unit tests because those fed
i.i.d.-random, mutually INDEPENDENT `dq` draws each tick (so
`eta_pred_m(k)` and `eta_pred_m(k-1)` were uncorrelated by test
construction, which a real trajectory never is) -- see
`test_scenario_c_autocorrelated_path_shows_old_bug_and_new_fix` below for
a direct, reproducible demonstration of the old bug and the fix.

**Scope restriction, stated plainly.** This derivation requires
`disturbance_filter_alpha == 0.0`. The constructor raises if a caller
configures anything else, rather than silently extending to a case this
derivation does not cover.

**What is identifiable and what is confounded -- demonstrated in
`test_identifiability_scenarios.py`, not just asserted here:**
- (A) constant bias, zero g: differencing cancels the constant bias
  exactly; g_hat correctly converges to ~0.
- (B) known nonzero g, zero bias, fresh excitation every tick: g_hat
  recovers g_true to within the RLS's own statistical tolerance.
- (C) changing reference/schedule index along the path, true g GLOBAL
  and constant: still correctly recovered (the global-g assumption
  holds by construction in this scenario) -- but the SAME scenario also
  demonstrates the old (pre-fix) code's systematic bias-toward-zero on
  autocorrelated motion, and that the new d_eta_pred_m(k)-based regressor does not
  share that failure mode.
- (D) changing/jumping bias between consecutive ticks: violates the
  "slowly varying" assumption directly -- demonstrated to corrupt g_hat
  for the duration of the jump; bounded (clipped), not silently
  undetected (the implausible-residual/implausible-dq gates catch large
  jumps, though a jump sized to pass both gates individually is NOT
  guaranteed to be caught -- stated as a real, open limitation, not
  hidden).
- (E) measurement timing/sync error (q_meas and the tip measurement
  drawn from different real instants despite both appearing "fresh" by
  the sample_age_s gate): demonstrated to corrupt g_hat without
  necessarily tripping any existing gate -- a genuine, undetected failure
  mode under the current per-sample age check alone. Documented as an
  open limitation, not claimed solved.

**Why a single scalar, not a free (3,6) or (3,7) correction**, and **why
insertion (column 6) is never adapted** (no independent sensor exists in
this rig): unchanged from the first version -- see the reasoning
preserved below.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

Array = np.ndarray

__all__ = ["JacobianAdaptationConfig", "ScalarJacobianAdaptation"]

_VALID_MODES = ("off", "shadow", "active")


@dataclass
class JacobianAdaptationConfig:
    mode: str = "off"
    # Bounds on the scalar correction g (dimensionless). Physically: the
    # beam's real joint-sensitivity is assumed to lie within
    # [1+g_min, 1+g_max] times the offline schedule's own value.
    g_min: float = -0.3
    g_max: float = 0.5
    # Forgetting factor for the recursive weighted least squares (0<lambda<=1;
    # 1.0 = no forgetting, pure running average; <1.0 lets g track a slow
    # drift and bounds the influence of very old samples).
    forgetting_lambda: float = 0.995
    # Minimum ||d_eta_pred_m(k)|| = ||eta_pred_m(k)-eta_pred_m(k')|| (meters) to accept a sample into
    # the regression -- "insufficient excitation" rejection, now correctly
    # gating on the DIFFERENCE (see module docstring: gating on ||eta_pred_m(k)||
    # alone was the old, wrong design -- a persistent, unchanging offset
    # has large ||eta_pred_m(k)|| but zero information about g).
    excitation_floor_m: float = 1.0e-4
    # Minimum accumulated excitation energy S(k) before g is allowed to
    # move away from 0 at all (guards the very first few ticks).
    min_excitation_energy: float = 1.0e-8
    # Reject a sample outright (don't incorporate, log reason) if the
    # instantaneous residual magnitude implies an implausible outlier.
    max_plausible_residual_m: float = 0.05
    # Reject if the joint state jumped further than this between
    # consecutive accepted samples (vision/encoder glitch indicator).
    max_plausible_dq_rad: float = 0.5
    # Reject (treat as stale) if the caller reports a measurement age above
    # this. None = no staleness check (caller did not provide an age).
    # NOTE (scope limitation, see module docstring scenario E): this
    # catches a GROSS single age being too old; it does NOT catch two
    # individually-fresh signals (joints, camera) drawn from DIFFERENT
    # real instants -- that failure mode is not covered by any gate here.
    max_sample_age_s: Optional[float] = 0.25
    # Reject duplicate camera frames (same frame_index as last accepted
    # sample) -- a frozen frame silently relabeled as a new measurement.
    reject_duplicate_frame_index: bool = True
    # Assumed per-axis measurement noise std (m) for a SINGLE sample
    # (before differencing) -- the reported variance accounts for the
    # differencing noise doubling (Var(d_noise) = 2*sigma^2) internally.
    measurement_noise_std_m: float = 3.0e-4

    def __post_init__(self) -> None:
        if self.mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}; got {self.mode!r}")
        if not (self.g_min < 0.0 <= self.g_max or self.g_min <= 0.0 < self.g_max):
            raise ValueError("g_min must be < 0 <= g_max (0 -- the nominal schedule -- must be reachable).")
        if not (0.0 < self.forgetting_lambda <= 1.0):
            raise ValueError("forgetting_lambda must be in (0, 1].")


class ScalarJacobianAdaptation:
    """Recursive scalar (weighted least-squares, first-differenced)
    estimator for `g`. Pure numpy, no I/O. See module docstring for the
    estimator equation and its explicit scope restriction to
    `disturbance_filter_alpha == 0.0`."""

    def __init__(self, config: JacobianAdaptationConfig, *, disturbance_filter_alpha: float = 0.0) -> None:
        if float(disturbance_filter_alpha) != 0.0:
            raise ValueError(
                "ScalarJacobianAdaptation's differencing derivation requires "
                "disturbance_filter_alpha == 0.0 (see module docstring) -- got "
                f"{disturbance_filter_alpha!r}. This is a real scope restriction, "
                "not an oversight: the project-wide default is 0.0 and no existing "
                "caller overrides it, but refusing explicitly here rather than "
                "silently producing an unvalidated estimate under a nonzero alpha."
            )
        self.config = config
        self._S = 0.0  # accumulated (forgetting-weighted) sum of ||d_eta_pred_m(k)||^2
        self._C = 0.0  # accumulated (forgetting-weighted) sum of d_eta_pred_m(k).dr(k)
        self._g_hat = 0.0
        self._last_frame_index: Optional[int] = None
        # last ACCEPTED raw sample, for differencing against (NOT
        # necessarily tick k-1 -- rejected ticks in between are skipped,
        # so corruption from a rejected sample never propagates forward).
        self._last_eta_pred_m: Optional[Array] = None
        self._last_r: Optional[Array] = None
        self.n_accepted = 0
        self.n_rejected = 0
        self.fallback_active = False

    def reset(self) -> None:
        """Explicit reset between runs -- required by the offline test
        suite (estimator state must not silently carry over between
        unrelated runs sharing one long-lived worker process)."""
        cfg = self.config
        self.__init__(cfg, disturbance_filter_alpha=0.0)

    def step(
        self, *, instantaneous_residual: Array, delta_q6: Array, j_nominal6: Array,
        frame_index: Optional[int] = None, sample_age_s: Optional[float] = None,
    ) -> dict[str, Any]:
        """One update, called with THIS tick's raw `_estimate_output_
        residual` instantaneous output (NOT its filtered/EMA output --
        differencing against the last accepted raw sample is this
        estimator's OWN job, done internally, so passing the EMA state in
        here would double-apply the cancellation).

        Returns a diagnostics dict always containing at least: mode,
        g_hat (post-update, bounded), g_used (0.0 unless mode=='active'),
        accepted (bool), reject_reason (str|None), n_accepted, n_rejected,
        d_eta_pred_m (the differenced-excitation magnitude actually used, or None
        if rejected before that point), dr_norm_m, variance,
        fallback_active. Never raises."""
        cfg = self.config
        try:
            instantaneous_residual = np.asarray(instantaneous_residual, dtype=float).reshape(3)
            delta_q6 = np.asarray(delta_q6, dtype=float).reshape(6)
            j_nominal6 = np.asarray(j_nominal6, dtype=float).reshape(3, 6)
        except Exception as exc:  # noqa: BLE001 -- must never raise, always report
            self.n_rejected += 1
            self.fallback_active = True
            return self._diag(accepted=False, reject_reason=f"shape_error: {exc!r}")

        if not (np.all(np.isfinite(instantaneous_residual)) and np.all(np.isfinite(delta_q6))
                and np.all(np.isfinite(j_nominal6))):
            self.n_rejected += 1
            self.fallback_active = True
            return self._diag(accepted=False, reject_reason="non_finite_input")

        if cfg.reject_duplicate_frame_index and frame_index is not None and frame_index == self._last_frame_index:
            self.n_rejected += 1
            return self._diag(accepted=False, reject_reason="duplicate_frame_index")

        if cfg.max_sample_age_s is not None and sample_age_s is not None and sample_age_s > cfg.max_sample_age_s:
            self.n_rejected += 1
            return self._diag(accepted=False, reject_reason=f"stale_sample(age={sample_age_s:.3f}s>{cfg.max_sample_age_s:.3f}s)")

        if float(np.max(np.abs(delta_q6))) > cfg.max_plausible_dq_rad:
            self.n_rejected += 1
            return self._diag(accepted=False, reject_reason="implausible_dq_jump")

        if float(np.linalg.norm(instantaneous_residual)) > cfg.max_plausible_residual_m:
            self.n_rejected += 1
            return self._diag(accepted=False, reject_reason="implausible_residual_outlier")

        eta_pred_m = j_nominal6 @ delta_q6  # (3,) meters -- see module docstring's "REGRESSOR NAME"

        if self._last_eta_pred_m is None:
            # First accepted sample ever: nothing to difference against yet.
            self._last_eta_pred_m = eta_pred_m.copy()
            self._last_r = instantaneous_residual.copy()
            self._last_frame_index = frame_index
            self.n_accepted += 1
            return self._diag(accepted=True, reject_reason=None, d_eta_pred_m=None, dr_norm_m=None, first_sample=True)

        d_eta_pred_vec_m = eta_pred_m - self._last_eta_pred_m  # (3,) meters
        dr = instantaneous_residual - self._last_r
        d_eta_pred_m = float(np.linalg.norm(d_eta_pred_vec_m))  # scalar magnitude, meters

        if d_eta_pred_m < cfg.excitation_floor_m:
            self.n_rejected += 1
            # Do NOT advance _last_eta_pred_m/_last_r here: differencing the NEXT
            # sample against this one (still fresh) is strictly better
            # than against an even-older one, and skipping the update
            # avoids accumulating a long chain of near-zero-d_eta_pred_m comparisons.
            return self._diag(
                accepted=False, reject_reason=f"insufficient_excitation(d_eta_pred_m={d_eta_pred_m:.2e}m)",
                d_eta_pred_m=d_eta_pred_m, dr_norm_m=float(np.linalg.norm(dr)),
            )

        lam = cfg.forgetting_lambda
        self._S = lam * self._S + float(np.dot(d_eta_pred_vec_m, d_eta_pred_vec_m))
        self._C = lam * self._C + float(np.dot(d_eta_pred_vec_m, dr))
        self.n_accepted += 1
        self._last_eta_pred_m = eta_pred_m.copy()
        self._last_r = instantaneous_residual.copy()
        self._last_frame_index = frame_index

        if self._S < cfg.min_excitation_energy:
            g_raw = 0.0
        else:
            g_raw = self._C / self._S

        if not np.isfinite(g_raw):
            self.fallback_active = True
            g_bounded = 0.0
        else:
            self.fallback_active = False
            g_bounded = float(np.clip(g_raw, cfg.g_min, cfg.g_max))
        self._g_hat = g_bounded

        return self._diag(
            accepted=True, reject_reason=None, d_eta_pred_m=d_eta_pred_m, dr_norm_m=float(np.linalg.norm(dr)),
        )

    def _diag(
        self, *, accepted: bool, reject_reason: Optional[str],
        d_eta_pred_m: Optional[float] = None, dr_norm_m: Optional[float] = None, first_sample: bool = False,
    ) -> dict[str, Any]:
        cfg = self.config
        g_hat = 0.0 if self.fallback_active else self._g_hat
        g_used = g_hat if cfg.mode == "active" else 0.0
        # Var(g_hat) ~= Var(d_noise)/S = 2*sigma^2/S -- the x2 accounts for
        # differencing two independent noise samples (see module docstring).
        variance = (
            None if self._S <= 0.0
            else float(2.0 * cfg.measurement_noise_std_m ** 2 / self._S)
        )
        return {
            "mode": cfg.mode,
            "g_hat": g_hat,
            "g_used": g_used,
            "accepted": accepted,
            "reject_reason": reject_reason,
            "first_sample": first_sample,
            "n_accepted": self.n_accepted,
            "n_rejected": self.n_rejected,
            "d_eta_pred_m": d_eta_pred_m,
            "dr_norm_m": dr_norm_m,
            "accumulated_excitation_energy": self._S,
            "variance": variance,
            "fallback_active": self.fallback_active,
        }
