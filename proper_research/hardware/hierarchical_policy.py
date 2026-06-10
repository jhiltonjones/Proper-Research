from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class HierarchicalMPCPolicyConfig:
    """
    Simple preventive hierarchical self-triggered MPC scheduler.

    Uses only five preventive metrics:

        1. Local curvature ahead
        2. Previous SQP relative step size
        3. Clearance
        4. Beam Hessian conditioning
        5. MPC Hessian conditioning

    Levels:
        0: LTI, rollout allowed to grow
        1: LTV one-shot, rollout capped
        2: SQP full, rollout = 1
    """

    enabled: bool = False

    rollout_min: int = 1
    rollout_max: int = 5
    ltv_rollout_cap: int = 3

    sqp_full_iters: int = 5

    # ------------------------------------------------------------
    # Preventive metric thresholds
    # ------------------------------------------------------------

    # 1. Curvature ahead [1/m]
    # Tune these based on your logged curvature scale.
    curvature_warn_1pm: float = 90.0
    curvature_bad_1pm: float = 190.0

    # 2. Previous SQP relative step
    # sqp_du_rel_final ~ 1 means approx 100% relative update.
    sqp_du_rel_warn: float = 0.5
    sqp_du_rel_bad: float = 2.0

    # 3. Clearance [mm]
    clearance_warn_mm: float = 1.5
    clearance_bad_mm: float = 1.0

    # 4. Beam Hessian conditioning
    beam_cond_warn: float = 5e3
    beam_cond_bad: float = 1e5

    # 5. MPC Hessian conditioning
    mpc_cond_warn: float = 5e7
    mpc_cond_bad: float = 1e8

    # Hysteresis: downgrade only after this many lower-risk decisions.
    downgrade_patience: int = 2


@dataclass
class HierarchicalMPCState:
    rollout_steps: int = 1
    previous_level: int = 0
    downgrade_counter: int = 0


@dataclass(frozen=True)
class HierarchicalMPCDecision:
    solver_mode: str
    rollout_steps: int
    N_sqp: int
    confidence: float
    risk: float
    level: int
    reason: str


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


def choose_hierarchical_mpc_action(
    *,
    info_pre: dict[str, Any],
    state: HierarchicalMPCState,
    cfg: HierarchicalMPCPolicyConfig,
) -> tuple[HierarchicalMPCDecision, HierarchicalMPCState]:
    """
    Choose solver fidelity and rollout length before solving/applying controls.
    """

    if not cfg.enabled:
        decision = HierarchicalMPCDecision(
            solver_mode="lti",
            rollout_steps=int(state.rollout_steps),
            N_sqp=1,
            confidence=1.0,
            risk=0.0,
            level=0,
            reason="hierarchy_disabled",
        )
        return decision, state

    desired_level, reason = classify_preventive_level(info_pre, cfg)

    previous_level = int(state.previous_level)
    rollout_current = int(
        np.clip(
            state.rollout_steps,
            cfg.rollout_min,
            cfg.rollout_max,
        )
    )

    downgrade_counter = int(state.downgrade_counter)

    # ------------------------------------------------------------
    # Hysteresis:
    #   - escalate immediately
    #   - downgrade only after several consecutive lower-risk decisions
    # ------------------------------------------------------------
    if desired_level > previous_level:
        level = desired_level
        downgrade_counter = 0
        reason += "|escalate"

    elif desired_level < previous_level:
        downgrade_counter += 1

        if downgrade_counter >= cfg.downgrade_patience:
            level = desired_level
            downgrade_counter = 0
            reason += "|downgrade_allowed"
        else:
            level = previous_level
            reason += "|downgrade_held"

    else:
        level = desired_level
        downgrade_counter = 0
        reason += "|hold"

    # ------------------------------------------------------------
    # Map level to controller complexity and rollout
    # ------------------------------------------------------------
    if level == 0:
        solver_mode = "lti"
        N_sqp = 1
        rollout_next = min(rollout_current + 1, cfg.rollout_max)
        confidence = 1.0
        risk = 0.0

    elif level == 1:
        solver_mode = "ltv_oneshot"
        N_sqp = 1
        rollout_next = min(max(rollout_current, cfg.rollout_min), cfg.ltv_rollout_cap)
        confidence = 0.5
        risk = 0.5

    else:
        solver_mode = "sqp_full"
        N_sqp = cfg.sqp_full_iters
        rollout_next = cfg.rollout_min
        confidence = 0.0
        risk = 1.0

    decision = HierarchicalMPCDecision(
        solver_mode=str(solver_mode),
        rollout_steps=int(rollout_next),
        N_sqp=int(N_sqp),
        confidence=float(confidence),
        risk=float(risk),
        level=int(level),
        reason=str(reason),
    )

    new_state = HierarchicalMPCState(
        rollout_steps=int(rollout_next),
        previous_level=int(level),
        downgrade_counter=int(downgrade_counter),
    )

    return decision, new_state