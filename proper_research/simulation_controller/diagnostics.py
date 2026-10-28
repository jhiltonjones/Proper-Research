import numpy as np

def classify_constraint_issue(diag):
    if int(diag.get("constraint_num_inconsistent_bounds", 0)) > 0:
        return "inconsistent_bounds_l_greater_than_u"

    block_names = [
        k.replace("constr_", "").replace("_max_vio_guess", "")
        for k in diag.keys()
        if k.startswith("constr_") and k.endswith("_max_vio_guess")
    ]

    if not block_names:
        return "no_constraint_blocks"

    worst_name = None
    worst_value = -np.inf

    for name in block_names:
        value = float(diag.get(f"constr_{name}_max_vio_guess", 0.0))
        if value > worst_value:
            worst_value = value
            worst_name = name

    if worst_value <= 1e-8:
        return "no_violation_at_U_guess"

    if worst_name == "input_bounds":
        return "input_bounds_violate_U_guess"

    if worst_name == "trust_region":
        return "trust_region_violate_U_guess"

    if worst_name == "dL_bounds":
        return "insertion_bounds_violate_U_guess"

    if "clearance" in worst_name:
        return "hard_clearance_violate_U_guess"

    return f"largest_violation_in_{worst_name}"
def diagnose_constraints_at_U(A, l, u, U, constraint_slices, *, tol=1e-8):
    """
    Diagnose l <= A U <= u at a candidate U.
    """
    A = np.asarray(A, float)
    l = np.asarray(l, float).reshape(-1)
    u = np.asarray(u, float).reshape(-1)
    U = np.asarray(U, float).reshape(-1)

    z = A @ U

    lower_violation = np.maximum(l - z, 0.0)
    upper_violation = np.maximum(z - u, 0.0)
    violation = np.maximum(lower_violation, upper_violation)

    inconsistent = l > u

    diag = {
        "constraint_max_vio_guess": float(np.nanmax(violation)) if violation.size else 0.0,
        "constraint_num_vio_guess": int(np.sum(violation > tol)),
        "constraint_num_inconsistent_bounds": int(np.sum(inconsistent)),
        "constraint_worst_row": int(np.nanargmax(violation)) if violation.size else -1,
        "constraint_worst_l": float(l[np.nanargmax(violation)]) if violation.size else np.nan,
        "constraint_worst_z": float(z[np.nanargmax(violation)]) if violation.size else np.nan,
        "constraint_worst_u": float(u[np.nanargmax(violation)]) if violation.size else np.nan,
    }

    for name, sl in constraint_slices.items():
        vv = violation[sl]
        ll = l[sl]
        zz = z[sl]
        uu = u[sl]
        bad_bounds = inconsistent[sl]

        if vv.size:
            j = int(np.nanargmax(vv))
            max_vio = float(vv[j])
            worst_l = float(ll[j])
            worst_z = float(zz[j])
            worst_u = float(uu[j])
        else:
            max_vio = 0.0
            worst_l = np.nan
            worst_z = np.nan
            worst_u = np.nan

        key = name.replace(" ", "_")

        diag[f"constr_{key}_max_vio_guess"] = max_vio
        diag[f"constr_{key}_num_vio_guess"] = int(np.sum(vv > tol))
        diag[f"constr_{key}_bad_bounds"] = int(np.sum(bad_bounds))
        diag[f"constr_{key}_worst_l"] = worst_l
        diag[f"constr_{key}_worst_z"] = worst_z
        diag[f"constr_{key}_worst_u"] = worst_u

    return diag
def qp_objective_value(H, f, z):
    """
    Evaluate OSQP-style quadratic objective:

        0.5 z.T H z + f.T z
    """
    H = np.asarray(H, float)
    f = np.asarray(f, float).reshape(-1)
    z = np.asarray(z, float).reshape(-1)

    return float(0.5 * z @ H @ z + f @ z)


def constraint_residuals(A, l, u, z, tol=1e-8):
    """
    Return constraint residual diagnostics for:

        l <= A z <= u
    """
    A = np.asarray(A, float)
    l = np.asarray(l, float).reshape(-1)
    u = np.asarray(u, float).reshape(-1)
    z = np.asarray(z, float).reshape(-1)

    Az = A @ z

    lower_violation = np.maximum(l - Az, 0.0)
    upper_violation = np.maximum(Az - u, 0.0)
    violation = np.maximum(lower_violation, upper_violation)

    finite_l = np.isfinite(l)
    finite_u = np.isfinite(u)

    return {
        "max_violation": float(np.max(violation)) if violation.size else 0.0,
        "num_violated": int(np.sum(violation > tol)),
        "lower_max_violation": float(np.max(lower_violation)) if lower_violation.size else 0.0,
        "upper_max_violation": float(np.max(upper_violation)) if upper_violation.size else 0.0,
        "Az_min": float(np.min(Az)) if Az.size else np.nan,
        "Az_max": float(np.max(Az)) if Az.size else np.nan,
        "finite_lower_count": int(np.sum(finite_l)),
        "finite_upper_count": int(np.sum(finite_u)),
    }


def prediction_tracking_metrics(X_pred, X_ref, n):
    """
    Compute tracking metrics between predicted and reference trajectories.

    X_pred:
        shape (Np, n) or (Np*n, 1)

    X_ref:
        shape (Np, n), (Np*n, 1), or (Np*n,)

    Returns position-only metrics for channels 0:3.
    """
    X_pred = np.asarray(X_pred, float)

    if X_pred.ndim == 1:
        if X_pred.size % n != 0:
            raise ValueError("X_pred size must be divisible by n.")
        X_pred = X_pred.reshape(-1, n)
    elif X_pred.ndim == 2 and X_pred.shape[1] == 1:
        if X_pred.size % n != 0:
            raise ValueError("X_pred size must be divisible by n.")
        X_pred = X_pred.reshape(-1, n)

    X_ref = np.asarray(X_ref, float)

    if X_ref.ndim == 1:
        if X_ref.size % n != 0:
            raise ValueError("X_ref size must be divisible by n.")
        X_ref = X_ref.reshape(-1, n)
    elif X_ref.ndim == 2 and X_ref.shape[1] == 1:
        if X_ref.size % n != 0:
            raise ValueError("X_ref size must be divisible by n.")
        X_ref = X_ref.reshape(-1, n)

    if X_pred.shape != X_ref.shape:
        raise ValueError(
            f"X_pred and X_ref shape mismatch: {X_pred.shape} vs {X_ref.shape}."
        )

    n_pos = min(3, n)

    err = X_pred[:, :n_pos] - X_ref[:, :n_pos]
    err_norm = np.linalg.norm(err, axis=1)

    return {
        "stage_error": err_norm.copy(),
        "rms_error": float(np.sqrt(np.mean(err_norm**2))),
        "max_error": float(np.max(err_norm)),
        "terminal_error": float(err_norm[-1]),
    }


def jacobian_svd_diagnostics(B, input_names=None, output_names=None):
    """
    SVD diagnostics for a local Jacobian B.

    Useful for checking local actuation authority.
    """
    B = np.asarray(B, float)

    if B.ndim != 2:
        raise ValueError("B must be a 2D matrix.")

    U, S, Vt = np.linalg.svd(B, full_matrices=False)

    if S.size == 0:
        cond = np.inf
        rank = 0
    else:
        cond = float(S[0] / max(S[-1], 1e-16))
        rank = int(np.sum(S > 1e-10))

    out = {
        "singular_values": S.copy(),
        "condition_number": cond,
        "rank": rank,
        "left_singular_vectors": U.copy(),
        "right_singular_vectors": Vt.copy(),
        "frobenius_norm": float(np.linalg.norm(B, "fro")),
        "column_norms": np.linalg.norm(B, axis=0),
        "row_norms": np.linalg.norm(B, axis=1),
    }

    if input_names is not None:
        input_names = list(input_names)

        if len(input_names) != B.shape[1]:
            raise ValueError("input_names length must match B.shape[1].")

        out["input_column_norms"] = {
            name: float(out["column_norms"][i])
            for i, name in enumerate(input_names)
        }

    if output_names is not None:
        output_names = list(output_names)

        if len(output_names) != B.shape[0]:
            raise ValueError("output_names length must match B.shape[0].")

        out["output_row_norms"] = {
            name: float(out["row_norms"][i])
            for i, name in enumerate(output_names)
        }

    return out


def print_jacobian_svd_summary(B, input_names=None, output_names=None, max_modes=3):
    """
    Print a compact SVD summary of B.
    """
    diag = jacobian_svd_diagnostics(
        B,
        input_names=input_names,
        output_names=output_names,
    )

    S = diag["singular_values"]
    Vt = diag["right_singular_vectors"]
    U = diag["left_singular_vectors"]

    print("[JAC] singular values:", np.array2string(S, precision=3))
    print(f"[JAC] rank={diag['rank']} cond={diag['condition_number']:.3e}")
    print(f"[JAC] ||B||_F={diag['frobenius_norm']:.3e}")

    if input_names is not None:
        print("[JAC] column norms:")
        for name, value in diag["input_column_norms"].items():
            print(f"  {name:>8s}: {value:.3e}")

    if output_names is not None:
        print("[JAC] row norms:")
        for name, value in diag["output_row_norms"].items():
            print(f"  {name:>8s}: {value:.3e}")

    max_modes = min(int(max_modes), Vt.shape[0])

    if input_names is not None:
        print("[JAC] dominant input directions:")
        for k in range(max_modes):
            entries = " ".join(
                f"{input_names[j]}:{Vt[k, j]:+.3f}"
                for j in range(Vt.shape[1])
            )
            print(f"  mode{k}: {entries}")

    if output_names is not None:
        print("[JAC] dominant output directions:")
        for k in range(max_modes):
            entries = " ".join(
                f"{output_names[i]}:{U[i, k]:+.3f}"
                for i in range(U.shape[0])
            )
            print(f"  mode{k}: {entries}")


def sqp_summary(sqp_hist):
    """
    Summarise SQP iteration history.
    """
    if sqp_hist is None or len(sqp_hist) == 0:
        return {
            "iterations": 0,
            "final_step_norm": np.nan,
            "final_rel_step_norm": np.nan,
            "statuses": [],
        }

    final = sqp_hist[-1]

    return {
        "iterations": len(sqp_hist),
        "final_step_norm": float(final.get("step_norm", np.nan)),
        "final_rel_step_norm": float(final.get("rel_step_norm", np.nan)),
        "statuses": [h.get("status", "unknown") for h in sqp_hist],
    }


def rollout_error_metrics(x_rollout, X_pred):
    """
    Compare actually applied rollout states with predicted states.

    x_rollout:
        shape (M, n), applied multi-step states

    X_pred:
        shape (Np, n), predicted horizon

    Compares first M predicted stages.
    """
    x_rollout = np.asarray(x_rollout, float)
    X_pred = np.asarray(X_pred, float)

    if x_rollout.ndim != 2:
        raise ValueError("x_rollout must have shape (M, n).")

    if X_pred.ndim != 2:
        raise ValueError("X_pred must have shape (Np, n).")

    M = x_rollout.shape[0]

    if X_pred.shape[0] < M:
        raise ValueError("X_pred has fewer stages than x_rollout.")

    if X_pred.shape[1] != x_rollout.shape[1]:
        raise ValueError(
            f"State dimension mismatch: {X_pred.shape[1]} vs {x_rollout.shape[1]}."
        )

    err = x_rollout - X_pred[:M, :]
    pos_err = err[:, :min(3, err.shape[1])]
    pos_norm = np.linalg.norm(pos_err, axis=1)

    return {
        "stage_position_error": pos_norm.copy(),
        "rms_position_error": float(np.sqrt(np.mean(pos_norm**2))),
        "max_position_error": float(np.max(pos_norm)),
        "terminal_position_error": float(pos_norm[-1]),
    }
import numpy as np


def beam_eigen_diagnostics(
    H_beam,
    *,
    cond_warn: float = 5e3,
    names=None,
) -> dict:
    """
    Diagnostic eigen-analysis for the beam Hessian.

    Intended for logging/analysis, not primary control switching.
    """
    if names is None:
        names = [
            "vx",
            "vy",
            "vz",
            "wx",
            "wy",
            "wz",
            "dL",
        ]

    out = {
        "beam_eig_logged": False,
        "beam_lambda_min": np.nan,
        "beam_lambda_max": np.nan,
        "beam_cond_from_eigs": np.nan,
        "beam_bad_direction_dominant_index": -1,
        "beam_bad_direction_dominant_name": "",
        "beam_bad_direction_norm": np.nan,
    }

    if H_beam is None:
        return out

    H = np.asarray(H_beam, float)

    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        return out

    # Symmetrise for numerical robustness.
    H = 0.5 * (H + H.T)

    try:
        eigvals, eigvecs = np.linalg.eigh(H)
    except np.linalg.LinAlgError:
        out["beam_eig_failed"] = True
        return out

    abs_eigs = np.abs(eigvals)
    finite = np.isfinite(abs_eigs)

    if not finite.any():
        return out

    idx_min = int(np.nanargmin(abs_eigs))
    idx_max = int(np.nanargmax(abs_eigs))

    lam_min = float(eigvals[idx_min])
    lam_max = float(eigvals[idx_max])

    cond = float(abs_eigs[idx_max] / max(abs_eigs[idx_min], 1e-12))

    out["beam_lambda_min"] = lam_min
    out["beam_lambda_max"] = lam_max
    out["beam_cond_from_eigs"] = cond

    if cond < cond_warn:
        return out

    v_min = eigvecs[:, idx_min]
    v_min = v_min / max(np.linalg.norm(v_min), 1e-12)

    dominant_index = int(np.argmax(np.abs(v_min)))
    dominant_name = names[dominant_index] if dominant_index < len(names) else str(dominant_index)

    out["beam_eig_logged"] = True
    out["beam_bad_direction_dominant_index"] = dominant_index
    out["beam_bad_direction_dominant_name"] = dominant_name
    out["beam_bad_direction_norm"] = float(np.linalg.norm(v_min))

    for j in range(min(len(v_min), len(names))):
        out[f"beam_bad_dir_{names[j]}"] = float(v_min[j])

    return out
def mpc_hessian_channel_diagnostics(
    H,
    *,
    Np: int,
    m: int = 7,
    cond_warn: float = 5e7,
    eps: float = 1e-12,
    control_names=("vx", "vy", "vz", "wx", "wy", "wz", "dL"),
) -> dict:
    out = {
        "mpc_eig_logged": False,
        "mpc_eig_failed": False,
        "mpc_eig_cond": np.nan,
        "mpc_weak_lambda": np.nan,
        "mpc_strong_lambda": np.nan,
        "mpc_weak_channel_index": -1,
        "mpc_weak_channel_name": "",
        "mpc_weak_channel_energy": np.nan,
        "mpc_strong_channel_index": -1,
        "mpc_strong_channel_name": "",
        "mpc_strong_channel_energy": np.nan,
    }

    for name in control_names:
        out[f"mpc_weak_energy_{name}"] = np.nan
        out[f"mpc_strong_energy_{name}"] = np.nan

    if H is None:
        return out

    H = np.asarray(H, float)

    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        out["mpc_eig_failed"] = True
        return out

    expected = int(Np) * int(m)

    if H.shape != (expected, expected):
        out["mpc_eig_failed"] = True
        out["mpc_H_shape_0"] = int(H.shape[0])
        out["mpc_H_shape_1"] = int(H.shape[1])
        out["mpc_H_expected"] = int(expected)
        return out

    if not np.all(np.isfinite(H)):
        out["mpc_eig_failed"] = True
        return out

    Hs = 0.5 * (H + H.T)

    try:
        eigvals, eigvecs = np.linalg.eigh(Hs)
    except np.linalg.LinAlgError:
        out["mpc_eig_failed"] = True
        return out

    abs_eigs = np.abs(eigvals)

    if not np.all(np.isfinite(abs_eigs)):
        out["mpc_eig_failed"] = True
        return out

    idx_min = int(np.argmin(abs_eigs))
    idx_max = int(np.argmax(abs_eigs))

    lam_min_abs = float(abs_eigs[idx_min])
    lam_max_abs = float(abs_eigs[idx_max])
    cond = float(lam_max_abs / max(lam_min_abs, eps))

    out["mpc_eig_cond"] = cond
    out["mpc_weak_lambda"] = float(eigvals[idx_min])
    out["mpc_strong_lambda"] = float(eigvals[idx_max])
    out["mpc_H_shape_0"] = int(H.shape[0])
    out["mpc_H_shape_1"] = int(H.shape[1])

    # Only interpret channel directions when conditioning is high.
    if cond < cond_warn:
        return out

    v_weak = eigvecs[:, idx_min]
    v_strong = eigvecs[:, idx_max]

    V_weak = v_weak.reshape(int(Np), int(m))
    V_strong = v_strong.reshape(int(Np), int(m))

    weak_energy = np.sum(V_weak**2, axis=0)
    strong_energy = np.sum(V_strong**2, axis=0)

    weak_energy = weak_energy / max(float(np.sum(weak_energy)), eps)
    strong_energy = strong_energy / max(float(np.sum(strong_energy)), eps)

    weak_idx = int(np.argmax(weak_energy))
    strong_idx = int(np.argmax(strong_energy))

    out["mpc_eig_logged"] = True

    out["mpc_weak_channel_index"] = weak_idx
    out["mpc_weak_channel_name"] = (
        control_names[weak_idx] if weak_idx < len(control_names) else str(weak_idx)
    )
    out["mpc_weak_channel_energy"] = float(weak_energy[weak_idx])

    out["mpc_strong_channel_index"] = strong_idx
    out["mpc_strong_channel_name"] = (
        control_names[strong_idx] if strong_idx < len(control_names) else str(strong_idx)
    )
    out["mpc_strong_channel_energy"] = float(strong_energy[strong_idx])

    for j, name in enumerate(control_names):
        if j < int(m):
            out[f"mpc_weak_energy_{name}"] = float(weak_energy[j])
            out[f"mpc_strong_energy_{name}"] = float(strong_energy[j])

    return out
def symmetric_matrix_diagnostics(H, *, eps=1e-12):
    """
    Return robust diagnostics for a symmetric Hessian-like matrix.

    Works for H_beam and H_mpc.
    """
    H = np.asarray(H, float)

    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        return {
            "valid": False,
            "cond": np.inf,
            "lambda_min": np.nan,
            "lambda_max": np.nan,
            "num_negative": -1,
            "num_near_zero": -1,
        }

    Hs = 0.5 * (H + H.T)

    try:
        eigvals = np.linalg.eigvalsh(Hs)
    except np.linalg.LinAlgError:
        return {
            "valid": False,
            "cond": np.inf,
            "lambda_min": np.nan,
            "lambda_max": np.nan,
            "num_negative": -1,
            "num_near_zero": -1,
        }

    abs_eigs = np.abs(eigvals)
    lam_min_abs = float(np.min(abs_eigs))
    lam_max_abs = float(np.max(abs_eigs))

    cond = lam_max_abs / max(lam_min_abs, eps)

    return {
        "valid": True,
        "cond": float(cond),
        "lambda_min": float(np.min(eigvals)),
        "lambda_max": float(np.max(eigvals)),
        "lambda_min_abs": lam_min_abs,
        "lambda_max_abs": lam_max_abs,
        "num_negative": int(np.sum(eigvals < -eps)),
        "num_near_zero": int(np.sum(abs_eigs < eps)),
    }