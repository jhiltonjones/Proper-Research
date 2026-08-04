from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class HierarchicalMPCPolicyConfig2:
    enabled: bool = False

    # Rollout limits.
    rollout_min: int = 1
    rollout_max: int = 5

    # Beam-Hessian risk thresholds:
    # risk_low_lambda_min_H_beam = -log10(abs(lambda_min_H_beam))
    beam_risk_low_max: float = 8.8665
    beam_risk_medium_max: float = 8.9859

    # Empirical rollout rule.
    low_risk_rollout: int = 4
    medium_risk_rollout: int = 2
    high_risk_rollout: int = 1

    # Use LTI only in low-risk mode.
    # For strict SQP statistical validation, set this False.
    use_lti_for_low_risk: bool = True
    lti_requires_previous_accepted: bool = True

    # Conservative default if beam risk is missing.
    missing_beam_risk_bin: str = "high"

    # SQP caps. These are caps, not acceptance criteria.
    sqp_cap_low: int = 4
    sqp_cap_medium: int = 4
    sqp_cap_high: int = 3
    sqp_cap_high_extension: int = 5

    # Final SQP tolerance rules.
    low_abs_tol: float = 2.4
    low_rel_guard: float = 2.0

    medium_abs_tol: float = 2.4
    medium_rel_guard: float = 2.0

    high_rel_tol: float = 0.60
    high_rel_strict_tol: float = 0.30
    high_abs_guard: float = 7.5

    # First-SQP-update probe thresholds.
    probe_abs_moderate: float = 3.35
    probe_abs_high: float = 5.89
    probe_rel_warn: float = 1.0
    probe_rel_high: float = 2.0

    # Secondary safety guards.
    clearance_stop_mm: float = 0.8
    clearance_bad_mm: float = 1.0
    clearance_warn_mm: float = 1.5

    # MPC conditioning is a numerical guard, not rollout selector.
    mpc_cond_extreme: float = 1e8

    # Previous prediction error guards.
    previous_adapt_error_bad_mm: float = 1.0
    previous_error_delta_bad_mm: float = 1.0

    # Hysteresis for returning to easy/LTI mode.
    downgrade_patience: int = 2

    # Eigenvalue floor for risk calculation.
    eigenvalue_floor: float = 1e-12

@dataclass
class HierarchicalMPCState:
    rollout_steps: int = 1
    previous_risk_bin: str = "unknown"
    previous_solver_mode: str = "sqp_full"
    previous_accepted: bool = False
    previous_level: int = 2
    downgrade_counter: int = 0
    last_reason: str = "initial"

@dataclass(frozen=True)
class HierarchicalMPCDecision:
    solver_mode: str
    rollout_steps: int
    N_sqp: int
    confidence: float
    risk: float
    level: int
    reason: str

    # Extra metadata.
    risk_bin: str = "unknown"
    beam_risk_value: float = np.nan
    rel_tol: float = np.nan
    abs_tol: float = np.nan
    strict_rel_tol: float = np.nan
    tolerance_mode: str = "none"
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
def _safe_float(info: dict[str, Any], key: str, default: float = np.nan) -> float:
    try:
        return float(info.get(key, default))
    except Exception:
        return float(default)


def _get_clearance_mm(info: dict[str, Any]) -> float:
    clearance = _safe_float(info, "clearance_mm", np.nan)

    if not np.isfinite(clearance):
        clearance = _safe_float(info, "adapt_clearance_mm", np.nan)

    return clearance


def _bad_previous_solver(info: dict[str, Any]) -> bool:
    status = str(info.get("status", info.get("previous_status", "solved")))
    infeasible = bool(info.get("infeasible", False))

    return infeasible or status not in (
        "solved",
        "solved_inaccurate",
        "solved inaccurate",
    )


def classify_preventive_level(
    info_pre: dict[str, Any],
    cfg: HierarchicalMPCPolicyConfig,
) -> tuple[int, str]:
    """
    Classify the current situation into one of three levels.

    Returns
    -------
    level:
        0 = easy, use LTI
        1 = warning, use LTV
        2 = hard/recovery, use SQP full

    reason:
        Human-readable trigger string.
    """

    curvature = _safe_float(info_pre, "curvature_max_ahead_1pm", np.nan)
    sqp_du_rel = _safe_float(info_pre, "sqp_du_rel_final", np.nan)
    clearance = _get_clearance_mm(info_pre)
    cond_beam = _safe_float(info_pre, "cond_H_beam", np.nan)
    cond_mpc = _safe_float(info_pre, "cond_H_mpc", np.nan)

    bad_reasons: list[str] = []
    warn_reasons: list[str] = []

    # ------------------------------------------------------------
    # Solver status safety override
    # ------------------------------------------------------------
    if _bad_previous_solver(info_pre):
        bad_reasons.append("previous_solver_failed")

    # ------------------------------------------------------------
    # 1. Curvature ahead
    # ------------------------------------------------------------
    if np.isfinite(curvature):
        if curvature >= cfg.curvature_bad_1pm:
            bad_reasons.append("curvature_bad")
        elif curvature >= cfg.curvature_warn_1pm:
            warn_reasons.append("curvature_warn")

    # ------------------------------------------------------------
    # 2. Previous SQP relative update
    # ------------------------------------------------------------
    if np.isfinite(sqp_du_rel):
        if sqp_du_rel >= cfg.sqp_du_rel_bad:
            bad_reasons.append("sqp_du_rel_bad")
        elif sqp_du_rel >= cfg.sqp_du_rel_warn:
            warn_reasons.append("sqp_du_rel_warn")

    # ------------------------------------------------------------
    # 3. Clearance
    # ------------------------------------------------------------
    if np.isfinite(clearance):
        if clearance <= cfg.clearance_bad_mm:
            bad_reasons.append("clearance_bad")
        elif clearance <= cfg.clearance_warn_mm:
            warn_reasons.append("clearance_warn")

    # ------------------------------------------------------------
    # 4. Beam Hessian conditioning
    # ------------------------------------------------------------
    if np.isfinite(cond_beam):
        if cond_beam >= cfg.beam_cond_bad:
            bad_reasons.append("beam_cond_bad")
        elif cond_beam >= cfg.beam_cond_warn:
            warn_reasons.append("beam_cond_warn")

    # ------------------------------------------------------------
    # 5. MPC Hessian conditioning
    # ------------------------------------------------------------
    if np.isfinite(cond_mpc):
        if cond_mpc >= cfg.mpc_cond_bad:
            bad_reasons.append("mpc_cond_bad")
        elif cond_mpc >= cfg.mpc_cond_warn:
            warn_reasons.append("mpc_cond_warn")

    # ------------------------------------------------------------
    # Level decision
    # ------------------------------------------------------------
    if bad_reasons:
        return 2, "hard:" + ",".join(bad_reasons)

    if warn_reasons:
        return 1, "moderate:" + ",".join(warn_reasons)

    return 0, "easy"
def evaluate_selected_policy_tolerance(
    *,
    info: dict[str, Any],
    decision: HierarchicalMPCDecision,
    cfg: HierarchicalMPCPolicyConfig,
) -> ToleranceEvaluation:

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
    if not np.isfinite(sqp_abs):
        sqp_abs = _safe_float(info, "sqp_du_abs_final", np.nan)

    sqp_rel = _safe_float(info, "sqp_du_rel_final", np.nan)

    rel_tol = float(decision.rel_tol)
    abs_tol = float(decision.abs_tol)
    strict_rel_tol = float(decision.strict_rel_tol)

    rel_ok = np.isfinite(sqp_rel) and np.isfinite(rel_tol) and sqp_rel <= rel_tol
    abs_ok = np.isfinite(sqp_abs) and np.isfinite(abs_tol) and sqp_abs <= abs_tol

    if decision.tolerance_mode == "abs_primary":
        accepted = bool(abs_ok and rel_ok)
        high_confidence = accepted

    elif decision.tolerance_mode == "rel_primary":
        accepted = bool(rel_ok and abs_ok)
        high_confidence = bool(
            accepted
            and np.isfinite(strict_rel_tol)
            and np.isfinite(sqp_rel)
            and sqp_rel <= strict_rel_tol
        )

    else:
        accepted = bool(abs_ok and rel_ok)
        high_confidence = accepted

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
    is_high_or_rollout1 = decision.risk_bin == "high" or rollout <= 1

    return ToleranceEvaluation(
        accepted=False,
        high_confidence=False,
        should_reduce_rollout=bool(can_reduce),
        should_continue_sqp=bool(is_high_or_rollout1),
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
        previous_risk_bin=str(decision.risk_bin),
        previous_solver_mode=str(decision.solver_mode),
        previous_accepted=bool(evaluation.accepted),
        previous_level=int(decision.level),
        downgrade_counter=int(state.downgrade_counter),
        last_reason=f"{decision.reason}|{evaluation.reason}",
    )
def choose_hierarchical_mpc_action(
    *,
    info_pre: dict[str, Any],
    state: HierarchicalMPCState,
    cfg: HierarchicalMPCPolicyConfig,
) -> tuple[HierarchicalMPCDecision, HierarchicalMPCState]:

    if not cfg.enabled:
        decision = HierarchicalMPCDecision(
            solver_mode="lti",
            rollout_steps=int(state.rollout_steps),
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
    rollout = candidate_rollout_for_risk(risk_bin, cfg)

    reasons = [
        f"beam_risk_bin={risk_bin}",
        f"beam_risk_source={risk_source}",
    ]

    # Previous solver failure: conservative override.
    if _bad_previous_solver(info_pre):
        risk_bin = "high"
        rollout = cfg.high_risk_rollout
        reasons.append("override:previous_solver_failed")

    # Clearance guards are secondary safety overrides.
    clearance = _safe_float(info_pre, "clearance_mm", np.nan)
    if not np.isfinite(clearance):
        clearance = _safe_float(info_pre, "adapt_clearance_mm", np.nan)

    if np.isfinite(clearance):
        if clearance <= cfg.clearance_stop_mm:
            risk_bin = "high"
            rollout = cfg.high_risk_rollout
            reasons.append("override:clearance_stop")
        elif clearance <= cfg.clearance_bad_mm:
            risk_bin = "high"
            rollout = min(rollout, 1)
            reasons.append("override:clearance_bad")
        elif clearance <= cfg.clearance_warn_mm:
            rollout = min(rollout, 2)
            if risk_bin == "low":
                risk_bin = "medium"
            reasons.append("guard:clearance_warn")

    # Previous prediction error guard.
    previous_adapt_err = _safe_float(info_pre, "adapt_pred_err_xy_mm", np.nan)
    if np.isfinite(previous_adapt_err) and previous_adapt_err >= cfg.previous_adapt_error_bad_mm:
        rollout = min(rollout, 2)
        if risk_bin == "low":
            risk_bin = "medium"
        reasons.append("guard:previous_adapt_error_bad")

    # MPC conditioning guard. This is not the primary rollout feature.
    cond_mpc = _safe_float(info_pre, "cond_H_mpc", np.nan)
    if not np.isfinite(cond_mpc):
        cond_mpc = _safe_float(info_pre, "mpc_eig_cond", np.nan)

    if np.isfinite(cond_mpc) and cond_mpc >= cfg.mpc_cond_extreme:
        rollout = min(rollout, 2)
        if risk_bin == "low":
            risk_bin = "medium"
        reasons.append("guard:mpc_cond_extreme")

    rollout = int(np.clip(rollout, cfg.rollout_min, cfg.rollout_max))

    # Optional LTI only in low risk.
    can_use_lti = (
        cfg.use_lti_for_low_risk
        and risk_bin == "low"
        and (
            bool(state.previous_accepted)
            or not cfg.lti_requires_previous_accepted
        )
    )

    if can_use_lti:
        solver_mode = "lti"
        N_sqp = 1
        rel_tol = np.nan
        abs_tol = np.nan
        strict_rel_tol = np.nan
        tolerance_mode = "none"
        level = 0
        reasons.append("solver:lti_low_risk")
    else:
        solver_mode = "sqp_full"
        N_sqp = sqp_cap_for_zone(risk_bin, rollout, cfg)
        rel_tol, abs_tol, strict_rel_tol, tolerance_mode = tolerance_for_zone(
            risk_bin,
            rollout,
            cfg,
        )
        level = 1 if risk_bin == "medium" else 2
        reasons.append("solver:sqp_full")

    risk_numeric = {
        "low": 0.15,
        "medium": 0.50,
        "high": 0.90,
        "unknown": 0.75,
    }.get(risk_bin, 0.75)

    confidence = 1.0 - risk_numeric

    decision = HierarchicalMPCDecision(
        solver_mode=str(solver_mode),
        rollout_steps=int(rollout),
        N_sqp=int(N_sqp),
        confidence=float(confidence),
        risk=float(risk_numeric),
        level=int(level),
        reason="|".join(reasons),
        risk_bin=str(risk_bin),
        beam_risk_value=float(beam_risk),
        rel_tol=float(rel_tol),
        abs_tol=float(abs_tol),
        strict_rel_tol=float(strict_rel_tol),
        tolerance_mode=str(tolerance_mode),
    )

    new_state = HierarchicalMPCState(
        rollout_steps=int(rollout),
        previous_risk_bin=str(risk_bin),
        previous_solver_mode=str(solver_mode),
        previous_accepted=bool(state.previous_accepted),
        previous_level=int(level),
        downgrade_counter=int(state.downgrade_counter),
        last_reason=str(decision.reason),
    )

    return decision, new_state