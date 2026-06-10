import numpy as np


def update_rollout_steps(
    *,
    current_rollout_steps: int,
    info: dict,
    rollout_min: int = 1,
    rollout_max: int = 3,
    cond_H_beam_bad: float = 1e10,
    cond_H_mpc_bad: float = 1e10,
    pred_err_bad_mm: float = 1.5,
    pred_err_good_mm: float = 0.85,
    clearance_bad_mm: float = 1.0,
    clearance_good_mm: float = 2.0,
    angle_bad_deg: float = 40.0,
    angle_good_deg: float = 30.0,
    stationarity_bad: float = 1e-3,
):
    r = int(current_rollout_steps)

    cond_H_beam = float(info.get("cond_H_beam", np.inf))
    cond_H_mpc = float(info.get("cond_H_mpc", np.inf))
    stationarity_norm = float(info.get("stationarity_norm", 0.0))

    pred_err = float(
        info.get(
            "pred1_err_xy_mm",
            info.get("rollout_err_xy_0_mm", np.inf),
        )
    )

    clearance = float(info.get("clearance_mm", np.inf))
    angle = float(info.get("tip_vessel_angle_deg", np.inf))

    infeasible = bool(info.get("infeasible", False))
    status = str(info.get("status", ""))

    num_neg_beam = int(info.get("num_negative_H_beam", 0))
    num_neg_mpc = int(info.get("num_negative_H_mpc", 0))

    bad_reasons = []

    if infeasible or status not in ("solved", "solved_inaccurate"):
        bad_reasons.append("solver")

    if not np.isfinite(cond_H_beam) or cond_H_beam > cond_H_beam_bad:
        bad_reasons.append("beam_hessian_condition")

    if not np.isfinite(cond_H_mpc) or cond_H_mpc > cond_H_mpc_bad:
        bad_reasons.append("mpc_hessian_condition")

    if num_neg_beam > 0:
        bad_reasons.append("beam_hessian_indefinite")

    if num_neg_mpc > 0:
        bad_reasons.append("mpc_hessian_indefinite")

    if np.isfinite(stationarity_norm) and stationarity_norm > stationarity_bad:
        bad_reasons.append("stationarity")

    if np.isfinite(pred_err) and pred_err > pred_err_bad_mm:
        bad_reasons.append("prediction_error")

    if np.isfinite(clearance) and clearance < clearance_bad_mm:
        bad_reasons.append("low_clearance")

    if np.isfinite(angle) and angle > angle_bad_deg:
        bad_reasons.append("tip_angle")

    if bad_reasons:
        return rollout_min, "bad:" + ",".join(bad_reasons)

    good = (
        np.isfinite(cond_H_beam)
        and cond_H_beam < 0.1 * cond_H_beam_bad
        and np.isfinite(cond_H_mpc)
        and cond_H_mpc < 0.1 * cond_H_mpc_bad
        and np.isfinite(stationarity_norm)
        and stationarity_norm < 0.1 * stationarity_bad
        and np.isfinite(pred_err)
        and pred_err < pred_err_good_mm
        and np.isfinite(clearance)
        and clearance > clearance_good_mm
        and np.isfinite(angle)
        and angle < angle_good_deg
    )

    if good:
        return min(r + 1, rollout_max), "good:increase"

    return r, "neutral:hold"