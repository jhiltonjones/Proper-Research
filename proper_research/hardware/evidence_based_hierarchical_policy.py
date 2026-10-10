from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

import math
import numpy as np

RiskBin = Literal["low", "medium", "high", "unknown"]
ToleranceMode = Literal["none", "abs_primary", "rel_primary"]


@dataclass(frozen=True)
class HierarchicalMPCPolicyConfig:
    """
    Evidence-based rollout/SQP scheduler from the statistical analysis.

    Controller hierarchy
    --------------------
    1. Beam-Hessian weak-direction risk chooses the maximum candidate rollout.
    2. Optional SQP-probe diagnostics can reduce, but never increase, rollout.
    3. Final SQP tolerance accepts/rejects the selected rollout.
    4. SQP iteration count is only a cap used to reach the tolerance.

    Important
    ---------
    The statistical policy was fitted mainly on SQP-full event data.  The LTI
    easy-mode option is included for integration with your hierarchy, but if you
    want the strictly evidence-matched policy, set:

        use_lti_for_low_risk = False

    Then low-risk also uses sqp_full with rollout 4 and the low-risk tolerance.
    """

    enabled: bool = False

    # ------------------------------------------------------------
    # Rollout limits and selected rollout rule
    # ------------------------------------------------------------
    rollout_min: int = 1
    rollout_max: int = 5

    # Empirical thresholds for
    #   risk_low_lambda_min_H_beam = -log10(abs(lambda_min_H_beam))
    # from the selected-policy analysis.
    beam_risk_low_max: float = 8.8665
    beam_risk_medium_max: float = 8.9859

    low_risk_rollout: int = 5
    medium_risk_rollout: int = 3
    high_risk_rollout: int = 1

    # If True, low-risk geometry uses the cheap LTI model.  If False, low-risk
    # still uses SQP full.  The latter is the strict version matching the SQP
    # tolerance statistics directly.
    use_lti_for_low_risk: bool = True

    # LTI easy-mode is safer if it requires a previously accepted solve.  This
    # prevents the first step, or a recovery step, from jumping directly into LTI.
    lti_requires_previous_accepted: bool = True

    # Missing beam-risk feature policy.  For hardware, conservative default is high.
    missing_beam_risk_bin: RiskBin = "high"

    # ------------------------------------------------------------
    # SQP caps: caps, not acceptance criteria
    # ------------------------------------------------------------
    sqp_cap_low: int = 7
    sqp_cap_medium: int = 5
    sqp_cap_high: int = 3

    # If the final tolerance is not met at rollout 1, optionally allow extra SQP
    # attempts before damping/holding.  The policy function reports this but does
    # not itself run extra SQP.
    sqp_cap_high_extension: int = 5

    # ------------------------------------------------------------
    # Final tolerance policy from selected-zone statistics
    # ------------------------------------------------------------
    # Low risk / rollout 4 and medium risk / rollout 2:
    # abs_final <= 2.4 was the balanced tolerance; rel <= 2.0 is a loose guard.
    low_abs_tol: float = 2.4
    low_rel_guard: float = 2.0
    medium_abs_tol: float = 2.4
    medium_rel_guard: float = 2.0

    # High risk / rollout 1:
    # relative tolerance was more useful than absolute tolerance.  abs <= 7.5 is
    # a broad guard; rel <= 0.30 is the strict high-confidence threshold.
    high_rel_tol: float = 0.60
    high_rel_strict_tol: float = 0.30
    high_abs_guard: float = 7.5

    # ------------------------------------------------------------
    # SQP probe/veto thresholds from threshold sweeps
    # ------------------------------------------------------------
    probe_abs_moderate: float = 3.35
    probe_abs_high: float = 5.89
    probe_rel_warn: float = 1.0
    probe_rel_high: float = 2.0

    # ------------------------------------------------------------
    # Secondary safety guards.  These are not the primary rollout policy.
    # ------------------------------------------------------------
    clearance_stop_mm: float = 0.8
    clearance_bad_mm: float = 1.0
    clearance_warn_mm: float = 1.5

    # MPC conditioning was weaker as a rollout selector, but can still be used
    # as a numerical-solve guard.
    mpc_cond_extreme: float = 1e8

    # Previous prediction error / spike guards.
    previous_adapt_error_bad_mm: float = 1.0
    previous_error_delta_bad_mm: float = 1.0

    # Hysteresis on returning to LTI/easier modes.
    downgrade_patience: int = 2

    # Numerical floor for eigenvalue-derived risk.
    eigenvalue_floor: float = 1e-12


@dataclass
class HierarchicalMPCState:
    rollout_steps: int = 1
    previous_risk_bin: RiskBin = "unknown"
    previous_solver_mode: str = "sqp_full"
    previous_accepted: bool = False
    previous_level: int = 2
    downgrade_counter: int = 0
    last_reason: str = "initial"


@dataclass(frozen=True)
class HierarchicalMPCDecision:
    solver_mode: str
    rollout_steps: int
    Np: int
    N_sqp: int
    confidence: float
    risk: float
    level: int
    reason: str

    # Evidence-policy metadata.
    risk_bin: RiskBin = "unknown"
    beam_risk_value: float = np.nan

    # Tolerance metadata.
    rel_tol: float = np.nan
    abs_tol: float = np.nan
    strict_rel_tol: float = np.nan
    tolerance_mode: ToleranceMode = "none"

    # Debug/probe metadata.
    probe_status: str = "not_run"
    accepted_by_tolerance: bool | None = None



@dataclass(frozen=True)
class ToleranceEvaluation:
    accepted: bool
    high_confidence: bool
    should_reduce_rollout: bool
    should_continue_sqp: bool
    should_damp_or_hold: bool
    reason: str
    sqp_du_abs_final: float = np.nan
    sqp_du_rel_final: float = np.nan
    rel_tol: float = np.nan
    abs_tol: float = np.nan
    strict_rel_tol: float = np.nan


# ---------------------------------------------------------------------------
# Generic safe extraction helpers
# ---------------------------------------------------------------------------

def _safe_float(info: dict[str, Any], key: str, default: float = np.nan) -> float:
    try:
        v = info.get(key, default)
        if v is None or v == "":
            return float(default)
        return float(v)
    except Exception:
        return float(default)

def prediction_horizon_for_zone(
    risk_bin: RiskBin,
    rollout: int,
    cfg: HierarchicalMPCPolicyConfig,
) -> int:
    """
    Evidence-based adaptive horizon.

    Since the Np/rollout analysis showed no clear benefit from planning beyond
    the executed rollout, use Np = selected rollout.
    """
    return _clip_rollout(int(rollout), cfg)
def _safe_bool(info: dict[str, Any], key: str, default: bool = False) -> bool:
    try:
        v = info.get(key, default)
        if isinstance(v, str):
            return v.strip().lower() in {"1", "true", "yes", "y"}
        return bool(v)
    except Exception:
        return bool(default)


def _is_finite(x: float) -> bool:
    return bool(np.isfinite(x))


def _clip_rollout(r: int, cfg: HierarchicalMPCPolicyConfig) -> int:
    return int(np.clip(int(r), int(cfg.rollout_min), int(cfg.rollout_max)))


def _bad_previous_solver(info: dict[str, Any]) -> bool:
    status = str(info.get("status", info.get("previous_status", "solved"))).strip().lower()
    infeasible = _safe_bool(info, "infeasible", False)
    return infeasible or status not in {"solved", "solved_inaccurate", "solved inaccurate"}


# ---------------------------------------------------------------------------
# Beam-Hessian risk extraction
# ---------------------------------------------------------------------------

def compute_beam_min_eigenvalue_risk(
    info: dict[str, Any],
    cfg: HierarchicalMPCPolicyConfig,
) -> tuple[float, str]:
    """
    Return risk_low_lambda_min_H_beam and the source key used.

    Preferred direct key:
        risk_low_lambda_min_H_beam

    Otherwise compute:
        -log10(max(abs(lambda_min), eigenvalue_floor))

    Accepted lambda-min keys cover the naming variants used in your simulation
    and hardware logs.
    """

    direct_keys = [
        "risk_low_lambda_min_H_beam",
        "beam_risk_low_lambda_min_H_beam",
    ]
    for key in direct_keys:
        value = _safe_float(info, key, np.nan)
        if _is_finite(value):
            return value, key

    lambda_keys = [
        "lambda_min_H_beam",
        "beam_lambda_min_H_beam",
        "beam_eig_lambda_min",
        "H_beam_lambda_min",
        "eig_lambda_min_H_beam",
    ]
    for key in lambda_keys:
        lam = _safe_float(info, key, np.nan)
        if _is_finite(lam):
            risk = -math.log10(max(abs(lam), float(cfg.eigenvalue_floor)))
            return float(risk), key

    return np.nan, "missing"


def classify_beam_risk(
    info: dict[str, Any],
    cfg: HierarchicalMPCPolicyConfig,
) -> tuple[RiskBin, float, str]:
    risk, source = compute_beam_min_eigenvalue_risk(info, cfg)

    if not _is_finite(risk):
        return cfg.missing_beam_risk_bin, np.nan, source

    if risk <= cfg.beam_risk_low_max:
        return "low", risk, source
    if risk <= cfg.beam_risk_medium_max:
        return "medium", risk, source
    return "high", risk, source


# ---------------------------------------------------------------------------
# Rollout/tolerance maps
# ---------------------------------------------------------------------------

def candidate_rollout_for_risk(
    risk_bin: RiskBin,
    cfg: HierarchicalMPCPolicyConfig,
) -> int:
    if risk_bin == "low":
        return _clip_rollout(cfg.low_risk_rollout, cfg)
    if risk_bin == "medium":
        return _clip_rollout(cfg.medium_risk_rollout, cfg)
    return _clip_rollout(cfg.high_risk_rollout, cfg)


def sqp_cap_for_zone(
    risk_bin: RiskBin,
    rollout: int,
    cfg: HierarchicalMPCPolicyConfig,
) -> int:
    if risk_bin == "low" and rollout >= 4:
        return int(cfg.sqp_cap_low)
    if risk_bin == "medium" and rollout >= 2:
        return int(cfg.sqp_cap_medium)
    if rollout <= 1 or risk_bin == "high":
        return int(cfg.sqp_cap_high)
    # Fallback after probe reductions.
    if rollout == 2:
        return int(cfg.sqp_cap_medium)
    return int(cfg.sqp_cap_low)


def tolerance_for_zone(
    risk_bin: RiskBin,
    rollout: int,
    cfg: HierarchicalMPCPolicyConfig,
) -> tuple[float, float, float, ToleranceMode]:
    """
    Returns rel_tol, abs_tol, strict_rel_tol, tolerance_mode.
    """
    if risk_bin == "low" and rollout >= 4:
        return cfg.low_rel_guard, cfg.low_abs_tol, np.nan, "abs_primary"

    if risk_bin == "medium" and rollout >= 2:
        return cfg.medium_rel_guard, cfg.medium_abs_tol, np.nan, "abs_primary"

    if risk_bin == "high" or rollout <= 1:
        return cfg.high_rel_tol, cfg.high_abs_guard, cfg.high_rel_strict_tol, "rel_primary"

    # Fallback if probe lowered rollout from low risk to 2/3.
    return cfg.medium_rel_guard, cfg.medium_abs_tol, np.nan, "abs_primary"


def _risk_numeric(risk_bin: RiskBin) -> float:
    return {"low": 0.15, "medium": 0.50, "high": 0.90, "unknown": 0.75}.get(risk_bin, 0.75)


def _level_for_decision(solver_mode: str, risk_bin: RiskBin, rollout: int) -> int:
    if solver_mode == "lti":
        return 0
    if risk_bin == "medium" or rollout == 2:
        return 1
    return 2


# ---------------------------------------------------------------------------
# Optional SQP probe / rollout veto
# ---------------------------------------------------------------------------

def extract_sqp_first_update(info: dict[str, Any]) -> tuple[float, float]:
    """Return first absolute and relative SQP update if present."""
    abs_keys = ["sqp_du_0", "sqp_du_first", "sqp_du_abs_first"]
    rel_keys = ["sqp_du_rel_0", "sqp_du_rel_first"]

    abs_first = np.nan
    rel_first = np.nan

    for key in abs_keys:
        value = _safe_float(info, key, np.nan)
        if _is_finite(value):
            abs_first = value
            break

    for key in rel_keys:
        value = _safe_float(info, key, np.nan)
        if _is_finite(value):
            rel_first = value
            break

    return abs_first, rel_first


def revise_rollout_after_sqp_probe(
    *,
    risk_bin: RiskBin,
    candidate_rollout: int,
    sqp_du_abs_first: float,
    sqp_du_rel_first: float,
    cfg: HierarchicalMPCPolicyConfig,
) -> tuple[int, str]:
    """
    Probe rule from the statistical threshold sweeps.

    The probe can only reduce rollout.  Absolute first update is primary;
    relative first update is secondary because relative-only changes can be
    denominator artefacts.
    """
    rollout = int(candidate_rollout)

    moderate_abs = _is_finite(sqp_du_abs_first) and sqp_du_abs_first >= cfg.probe_abs_moderate
    high_abs = _is_finite(sqp_du_abs_first) and sqp_du_abs_first >= cfg.probe_abs_high
    rel_warn = _is_finite(sqp_du_rel_first) and sqp_du_rel_first >= cfg.probe_rel_warn
    rel_high = _is_finite(sqp_du_rel_first) and sqp_du_rel_first >= cfg.probe_rel_high

    if high_abs and rel_warn:
        return _clip_rollout(1, cfg), "probe_reduce_to_1:high_abs_and_rel"

    if high_abs:
        return _clip_rollout(min(rollout, 2), cfg), "probe_reduce_to_2:high_abs"

    if moderate_abs and rel_high:
        return _clip_rollout(min(rollout, 2), cfg), "probe_reduce_to_2:moderate_abs_high_rel"

    if moderate_abs and rel_warn:
        return _clip_rollout(min(rollout, 3), cfg), "probe_reduce_to_3:moderate_abs_rel"

    if rel_high and not moderate_abs:
        return _clip_rollout(min(rollout, 3), cfg), "probe_relative_only_warning"

    return _clip_rollout(rollout, cfg), "probe_ok"


# ---------------------------------------------------------------------------
# Main pre-solve decision function: preserves your existing API
# ---------------------------------------------------------------------------

def choose_hierarchical_mpc_action(
    *,
    info_pre: dict[str, Any],
    state: HierarchicalMPCState,
    cfg: HierarchicalMPCPolicyConfig,
) -> tuple[HierarchicalMPCDecision, HierarchicalMPCState]:
    """
    Pre-solve decision.

    This preserves the existing call signature used by both hardware and
    simulation code.  It chooses a candidate rollout and solver/SQP cap using
    the beam-Hessian risk feature.  Final tolerance acceptance is evaluated
    after mpc.step(...) using evaluate_selected_policy_tolerance(...).
    """

    if not cfg.enabled:
        rollout = _clip_rollout(state.rollout_steps, cfg)
        decision = HierarchicalMPCDecision(
            solver_mode="lti",
            rollout_steps=rollout,
            Np=rollout,
            N_sqp=1,
            confidence=1.0,
            risk=0.0,
            level=0,
            reason="hierarchy_disabled",
            risk_bin="low",
            tolerance_mode="none",
        )
        return decision, state

    risk_bin, beam_risk, risk_source = classify_beam_risk(info_pre, cfg)
    candidate_rollout = candidate_rollout_for_risk(risk_bin, cfg)

    reasons: list[str] = [
        f"beam_risk_bin={risk_bin}",
        f"beam_risk_source={risk_source}",
    ]

    # Conservative overrides from previous status / hard safety metrics.
    if _bad_previous_solver(info_pre):
        risk_bin = "high"
        candidate_rollout = cfg.high_risk_rollout
        reasons.append("override:previous_solver_failed")

    clearance = _safe_float(info_pre, "clearance_mm", np.nan)
    if not _is_finite(clearance):
        clearance = _safe_float(info_pre, "adapt_clearance_mm", np.nan)

    if _is_finite(clearance):
        if clearance <= cfg.clearance_stop_mm:
            risk_bin = "high"
            candidate_rollout = cfg.high_risk_rollout
            reasons.append("override:clearance_stop")
        elif clearance <= cfg.clearance_bad_mm:
            candidate_rollout = min(candidate_rollout, 1)
            risk_bin = "high"
            reasons.append("override:clearance_bad")
        elif clearance <= cfg.clearance_warn_mm:
            candidate_rollout = min(candidate_rollout, 2)
            if risk_bin == "low":
                risk_bin = "medium"
            reasons.append("guard:clearance_warn")

    previous_adapt_err = _safe_float(info_pre, "adapt_pred_err_xy_mm", np.nan)
    if _is_finite(previous_adapt_err) and previous_adapt_err >= cfg.previous_adapt_error_bad_mm:
        candidate_rollout = min(candidate_rollout, 2)
        if risk_bin == "low":
            risk_bin = "medium"
        reasons.append("guard:previous_adapt_error_bad")

    previous_delta = _safe_float(info_pre, "next_error_delta", np.nan)
    if _is_finite(previous_delta) and previous_delta >= cfg.previous_error_delta_bad_mm:
        candidate_rollout = min(candidate_rollout, 1)
        risk_bin = "high"
        reasons.append("override:previous_error_spike")

    cond_mpc = _safe_float(info_pre, "cond_H_mpc", np.nan)
    if not _is_finite(cond_mpc):
        cond_mpc = _safe_float(info_pre, "mpc_eig_cond", np.nan)
    if _is_finite(cond_mpc) and cond_mpc >= cfg.mpc_cond_extreme:
        # MPC conditioning was not the primary rollout selector, but extreme
        # numerical conditioning should prevent cheap/easy execution.
        candidate_rollout = min(candidate_rollout, 2)
        if risk_bin == "low":
            risk_bin = "medium"
        reasons.append("guard:mpc_cond_extreme")

    candidate_rollout = _clip_rollout(candidate_rollout, cfg)

    # LTI is only used as low-risk/easy mode.  The strict evidence-matched SQP
    # controller is obtained by setting use_lti_for_low_risk=False.
    previous_ok = bool(state.previous_accepted)
    can_use_lti = (
        cfg.use_lti_for_low_risk
        and risk_bin == "low"
        and (previous_ok or not cfg.lti_requires_previous_accepted)
    )

    if can_use_lti:
        solver_mode = "lti"
        N_sqp = 1
        rel_tol = np.nan
        abs_tol = np.nan
        strict_rel_tol = np.nan
        tolerance_mode: ToleranceMode = "none"
        reasons.append("solver:lti_low_risk")
    else:
        solver_mode = "sqp_full"
        N_sqp = sqp_cap_for_zone(risk_bin, candidate_rollout, cfg)
        rel_tol, abs_tol, strict_rel_tol, tolerance_mode = tolerance_for_zone(
            risk_bin,
            candidate_rollout,
            cfg,
        )
        reasons.append("solver:sqp_full")

    level = _level_for_decision(solver_mode, risk_bin, candidate_rollout)

    # Downgrade hysteresis only prevents immediate return to LTI/low level.
    downgrade_counter = int(state.downgrade_counter)
    previous_level = int(state.previous_level)
    desired_level = int(level)

    if desired_level < previous_level:
        downgrade_counter += 1
        if downgrade_counter < cfg.downgrade_patience:
            # Hold previous harder mode for one more planning decision.
            desired_level = previous_level
            if previous_level >= 1 and solver_mode == "lti":
                solver_mode = "sqp_full"
                N_sqp = sqp_cap_for_zone("medium", min(candidate_rollout, 2), cfg)
                candidate_rollout = min(candidate_rollout, 2)
                rel_tol, abs_tol, strict_rel_tol, tolerance_mode = tolerance_for_zone(
                    "medium",
                    candidate_rollout,
                    cfg,
                )
                reasons.append("downgrade_held:force_sqp")
            else:
                reasons.append("downgrade_held")
        else:
            downgrade_counter = 0
            reasons.append("downgrade_allowed")
    else:
        downgrade_counter = 0
        reasons.append("hold_or_escalate")

    confidence = 1.0 - _risk_numeric(risk_bin)
    risk_numeric = _risk_numeric(risk_bin)
    Np = prediction_horizon_for_zone(
        risk_bin,
        candidate_rollout,
        cfg,
    )
    decision = HierarchicalMPCDecision(
        solver_mode=str(solver_mode),
        rollout_steps=int(candidate_rollout),
        Np=int(Np),
        N_sqp=int(N_sqp),
        confidence=float(confidence),
        risk=float(risk_numeric),
        level=int(desired_level),
        reason="|".join(reasons),
        risk_bin=risk_bin,
        beam_risk_value=float(beam_risk),
        rel_tol=float(rel_tol),
        abs_tol=float(abs_tol),
        strict_rel_tol=float(strict_rel_tol),
        tolerance_mode=tolerance_mode,
    )

    new_state = HierarchicalMPCState(
        rollout_steps=int(candidate_rollout),
        previous_risk_bin=risk_bin,
        previous_solver_mode=str(solver_mode),
        previous_accepted=bool(state.previous_accepted),
        previous_level=int(desired_level),
        downgrade_counter=int(downgrade_counter),
        last_reason=str(decision.reason),
    )

    return decision, new_state


# ---------------------------------------------------------------------------
# Post-solve tolerance acceptance
# ---------------------------------------------------------------------------

def evaluate_selected_policy_tolerance(
    *,
    info: dict[str, Any],
    decision: HierarchicalMPCDecision,
    cfg: HierarchicalMPCPolicyConfig,
) -> ToleranceEvaluation:
    """
    Evaluate whether the just-computed plan is acceptable.

    Call this immediately after mpc.step(...).  For LTI plans, no SQP tolerance
    is available, so acceptance is based on solver status only.  For SQP plans,
    use the zone-dependent final SQP tolerance.
    """
    if _bad_previous_solver(info):
        return ToleranceEvaluation(
            accepted=False,
            high_confidence=False,
            should_reduce_rollout=int(decision.rollout_steps) > 1,
            should_continue_sqp=False,
            should_damp_or_hold=int(decision.rollout_steps) <= 1,
            reason="reject:solver_failed_or_infeasible",
            rel_tol=decision.rel_tol,
            abs_tol=decision.abs_tol,
            strict_rel_tol=decision.strict_rel_tol,
        )

    if decision.solver_mode == "lti" or decision.tolerance_mode == "none":
        return ToleranceEvaluation(
            accepted=True,
            high_confidence=True,
            should_reduce_rollout=False,
            should_continue_sqp=False,
            should_damp_or_hold=False,
            reason="accept:lti_no_sqp_tolerance",
            rel_tol=np.nan,
            abs_tol=np.nan,
            strict_rel_tol=np.nan,
        )

    sqp_abs = _safe_float(info, "sqp_du_final", np.nan)
    if not _is_finite(sqp_abs):
        sqp_abs = _safe_float(info, "sqp_du_abs_final", np.nan)

    sqp_rel = _safe_float(info, "sqp_du_rel_final", np.nan)

    rel_tol = float(decision.rel_tol)
    abs_tol = float(decision.abs_tol)
    strict_rel_tol = float(decision.strict_rel_tol)

    rel_ok = _is_finite(sqp_rel) and _is_finite(rel_tol) and sqp_rel <= rel_tol
    abs_ok = _is_finite(sqp_abs) and _is_finite(abs_tol) and sqp_abs <= abs_tol

    if decision.tolerance_mode == "abs_primary":
        accepted = bool(abs_ok and rel_ok)
        high_confidence = bool(accepted)
    elif decision.tolerance_mode == "rel_primary":
        accepted = bool(rel_ok and abs_ok)
        high_confidence = bool(
            accepted
            and _is_finite(strict_rel_tol)
            and _is_finite(sqp_rel)
            and sqp_rel <= strict_rel_tol
        )
    else:
        accepted = bool(abs_ok and rel_ok)
        high_confidence = bool(accepted)

    if accepted:
        reason = "accept:tolerance_met"
        if high_confidence:
            reason += ":high_confidence"
        return ToleranceEvaluation(
            accepted=True,
            high_confidence=high_confidence,
            should_reduce_rollout=False,
            should_continue_sqp=False,
            should_damp_or_hold=False,
            reason=reason,
            sqp_du_abs_final=float(sqp_abs),
            sqp_du_rel_final=float(sqp_rel),
            rel_tol=rel_tol,
            abs_tol=abs_tol,
            strict_rel_tol=strict_rel_tol,
        )

    rollout = int(decision.rollout_steps)
    can_reduce = rollout > 1
    is_high_rollout1 = decision.risk_bin == "high" or rollout <= 1

    return ToleranceEvaluation(
        accepted=False,
        high_confidence=False,
        should_reduce_rollout=bool(can_reduce),
        should_continue_sqp=bool(is_high_rollout1),
        should_damp_or_hold=bool(not can_reduce),
        reason=(
            "reject:tolerance_not_met"
            f":abs={sqp_abs:.4g}<=?{abs_tol:.4g}"
            f":rel={sqp_rel:.4g}<=?{rel_tol:.4g}"
        ),
        sqp_du_abs_final=float(sqp_abs),
        sqp_du_rel_final=float(sqp_rel),
        rel_tol=rel_tol,
        abs_tol=abs_tol,
        strict_rel_tol=strict_rel_tol,
    )


def update_hierarchical_state_after_solve(
    *,
    state: HierarchicalMPCState,
    decision: HierarchicalMPCDecision,
    evaluation: ToleranceEvaluation,
) -> HierarchicalMPCState:
    return HierarchicalMPCState(
        rollout_steps=int(decision.rollout_steps),
        previous_risk_bin=decision.risk_bin,
        previous_solver_mode=decision.solver_mode,
        previous_accepted=bool(evaluation.accepted),
        previous_level=int(decision.level),
        downgrade_counter=int(state.downgrade_counter),
        last_reason=f"{decision.reason}|{evaluation.reason}",
    )


def attach_policy_metadata(
    info: dict[str, Any],
    *,
    decision: HierarchicalMPCDecision,
    evaluation: ToleranceEvaluation | None = None,
) -> dict[str, Any]:
    """Attach policy metadata to an MPC info dict for CSV logging."""
    info["hierarchy_enabled"] = True
    info["hierarchy_solver_mode"] = decision.solver_mode
    info["hierarchy_rollout_steps"] = int(decision.rollout_steps)
    info["hierarchy_N_sqp"] = int(decision.N_sqp)
    info["hierarchy_Np"] = int(decision.Np)
    info["Np_used"] = int(decision.Np)
    info["hierarchy_confidence"] = float(decision.confidence)
    info["hierarchy_risk"] = float(decision.risk)
    info["hierarchy_level"] = int(decision.level)
    info["hierarchy_reason"] = str(decision.reason)

    info["hierarchy_risk_bin"] = str(decision.risk_bin)
    info["hierarchy_beam_risk_value"] = float(decision.beam_risk_value)
    info["hierarchy_rel_tol"] = float(decision.rel_tol)
    info["hierarchy_abs_tol"] = float(decision.abs_tol)
    info["hierarchy_strict_rel_tol"] = float(decision.strict_rel_tol)
    info["hierarchy_tolerance_mode"] = str(decision.tolerance_mode)

    if evaluation is not None:
        info["hierarchy_tolerance_accepted"] = bool(evaluation.accepted)
        info["hierarchy_tolerance_high_confidence"] = bool(evaluation.high_confidence)
        info["hierarchy_tolerance_reason"] = str(evaluation.reason)
        info["hierarchy_should_reduce_rollout"] = bool(evaluation.should_reduce_rollout)
        info["hierarchy_should_continue_sqp"] = bool(evaluation.should_continue_sqp)
        info["hierarchy_should_damp_or_hold"] = bool(evaluation.should_damp_or_hold)

    return info
