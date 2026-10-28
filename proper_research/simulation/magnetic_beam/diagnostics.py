from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class FiniteDifferenceOptions:
    eps: float = 1e-5
    central: bool = True

    def validate(self) -> None:
        if self.eps <= 0:
            raise ValueError(f"eps must be positive, got {self.eps}.")


def finite_difference_jacobian(
    fun: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    *,
    options: FiniteDifferenceOptions | None = None,
) -> np.ndarray:
    if options is None:
        options = FiniteDifferenceOptions()
    options.validate()

    x0 = np.asarray(x0, float).reshape(-1)

    if not np.all(np.isfinite(x0)):
        raise ValueError(f"x0 contains non-finite values: {x0}.")

    y0 = np.asarray(fun(x0), float).reshape(-1)

    if not np.all(np.isfinite(y0)):
        raise ValueError(f"fun(x0) returned non-finite values: {y0}.")

    m = y0.size
    n = x0.size

    J = np.empty((m, n), dtype=float)

    for j in range(n):
        xp = x0.copy()
        xp[j] += options.eps

        if options.central:
            xm = x0.copy()
            xm[j] -= options.eps

            yp = np.asarray(fun(xp), float).reshape(-1)
            ym = np.asarray(fun(xm), float).reshape(-1)

            J[:, j] = (yp - ym) / (2.0 * options.eps)
        else:
            yp = np.asarray(fun(xp), float).reshape(-1)
            J[:, j] = (yp - y0) / options.eps

    return J


def compare_jacobians(
    J_ref: np.ndarray,
    J_test: np.ndarray,
    *,
    name_ref: str = "reference",
    name_test: str = "test",
    print_report: bool = True,
) -> dict:
    """
    Compare two Jacobians and return numerical diagnostics.
    """
    J_ref = np.asarray(J_ref, float)
    J_test = np.asarray(J_test, float)

    if J_ref.shape != J_test.shape:
        raise ValueError(
            f"Jacobian shapes do not match: {name_ref} {J_ref.shape}, "
            f"{name_test} {J_test.shape}."
        )

    err = J_test - J_ref

    abs_err = float(np.linalg.norm(err))
    ref_norm = float(np.linalg.norm(J_ref))
    test_norm = float(np.linalg.norm(J_test))
    rel_err = abs_err / (ref_norm + 1e-12)

    max_abs_err = float(np.max(np.abs(err)))
    col_abs_err = np.linalg.norm(err, axis=0)
    col_ref_norm = np.linalg.norm(J_ref, axis=0)
    col_rel_err = col_abs_err / (col_ref_norm + 1e-12)

    dot = float(np.sum(J_ref * J_test))
    cos = dot / ((ref_norm * test_norm) + 1e-12)

    report = {
        "abs_err": abs_err,
        "rel_err": float(rel_err),
        "max_abs_err": max_abs_err,
        "ref_norm": ref_norm,
        "test_norm": test_norm,
        "cosine": float(cos),
        "col_abs_err": col_abs_err,
        "col_rel_err": col_rel_err,
    }

    if print_report:
        print("\n--- JACOBIAN COMPARISON ---")
        print(f"{name_ref} norm = {ref_norm:.6e}")
        print(f"{name_test} norm = {test_norm:.6e}")
        print(f"abs err      = {abs_err:.6e}")
        print(f"rel err      = {rel_err:.6e}")
        print(f"max abs err  = {max_abs_err:.6e}")
        print(f"cosine       = {cos:.6e}")
        print("column relative errors:")
        for j, val in enumerate(col_rel_err):
            print(f"  col {j}: {val:.6e}")

    return report


def check_forward_jacobian_pose7(
    model,
    p7: np.ndarray,
    *,
    eps: float = 1e-5,
    solve_if_needed: bool = True,
    print_report: bool = True,
    restore_cache: bool = True,
) -> dict:
    """
    Compare model.jacobian_tip_pose7(p7) against finite differences of model(p7).

    By default this restores the model cache after the diagnostic.
    """
    p7 = np.asarray(p7, float).reshape(7)

    original_cache = model.get_cache_copy()

    try:
        # Commit nominal solve so analytic Jacobian uses matching cached u*.
        model.solve(p7, commit=True)

        J_analytic = model.jacobian_tip_pose7(
            p7,
            solve_if_needed=solve_if_needed,
        )

        base_cache = model.get_cache_copy()

        def f_pose7(x: np.ndarray) -> np.ndarray:
            old_cache = model.get_cache_copy()
            try:
                model.set_cache(base_cache)
                return model.solve(x, commit=True).tip.copy()
            finally:
                model.set_cache(old_cache)

        J_fd = finite_difference_jacobian(
            f_pose7,
            p7,
            options=FiniteDifferenceOptions(eps=eps, central=True),
        )

        report = compare_jacobians(
            J_ref=J_fd,
            J_test=J_analytic,
            name_ref="finite-difference",
            name_test="analytic",
            print_report=print_report,
        )

        report["J_fd"] = J_fd
        report["J_analytic"] = J_analytic

        return report

    finally:
        if restore_cache:
            model.set_cache(original_cache)


def check_gradient_against_energy(
    *,
    energy_fun: Callable[[np.ndarray], float],
    grad_fun: Callable[[np.ndarray], np.ndarray],
    u0: np.ndarray,
    eps: float = 1e-6,
    max_cols: int | None = 12,
    print_report: bool = True,
) -> dict:
    """
    Direction/coordinate check for dE/du.

    This is useful after changing magnetic/contact gradient code.
    """
    u0 = np.asarray(u0, float).reshape(-1)
    g = np.asarray(grad_fun(u0), float).reshape(-1)

    n = u0.size
    cols = range(n) if max_cols is None else range(min(n, max_cols))

    fd = []
    analytic = []
    abs_errs = []
    rel_errs = []

    for j in cols:
        up = u0.copy()
        um = u0.copy()

        up[j] += eps
        um[j] -= eps

        dE_fd = (float(energy_fun(up)) - float(energy_fun(um))) / (2.0 * eps)
        dE_an = float(g[j])

        err = dE_an - dE_fd
        rel = abs(err) / (abs(dE_fd) + 1e-12)

        fd.append(dE_fd)
        analytic.append(dE_an)
        abs_errs.append(abs(err))
        rel_errs.append(rel)

    fd = np.asarray(fd, float)
    analytic = np.asarray(analytic, float)
    abs_errs = np.asarray(abs_errs, float)
    rel_errs = np.asarray(rel_errs, float)

    report = {
        "fd": fd,
        "analytic": analytic,
        "abs_errs": abs_errs,
        "rel_errs": rel_errs,
        "max_abs_err": float(np.max(abs_errs)) if abs_errs.size else np.nan,
        "max_rel_err": float(np.max(rel_errs)) if rel_errs.size else np.nan,
        "grad_norm": float(np.linalg.norm(g)),
    }

    if print_report:
        print("\n--- GRADIENT CHECK ---")
        print(f"||grad||    = {report['grad_norm']:.6e}")
        print(f"max abs err = {report['max_abs_err']:.6e}")
        print(f"max rel err = {report['max_rel_err']:.6e}")
        for i, j in enumerate(cols):
            print(
                f"  u[{j:03d}]  "
                f"fd={fd[i]:+.6e}  "
                f"analytic={analytic[i]:+.6e}  "
                f"rel={rel_errs[i]:.6e}"
            )

    return report


def summarise_solve_result(result, *, prefix: str = "[SOLVE]") -> None:
    """
    Lightweight print summary for a SolveResult.
    """
    info = result.info
    parts = info.get("parts", {})

    print(prefix)
    print(f"  success: {info.get('success')}")
    print(f"  optimizer success: {info.get('optimizer_success')}")
    print(
        "  accepted stationary failure: "
        f"{info.get('accepted_stationary_failure')}"
    )
    print(f"  message: {info.get('message')}")
    print(f"  nit:     {info.get('nit')}")
    print(f"  nfev:    {info.get('nfev')}")
    print(f"  njev:    {info.get('njev')}")
    print(
        "  ||grad_z||_inf: "
        f"{info.get('grad_inf_norm_scaled', np.nan):.6e}"
    )
    print(
        "  gradient-check max relative error: "
        f"{(info.get('gradient_check') or {}).get('max_relative_error', np.nan):.6e}"
    )
    print(f"  W:       {info.get('W', np.nan):.6e}")
    print(f"  dW:      {info.get('dW', np.nan):.6e}")
    print(f"  W_el:    {parts.get('W_el', np.nan):.6e}")
    print(f"  W_m:     {parts.get('W_m', np.nan):.6e}")
    print(f"  W_cf:    {parts.get('W_cf', np.nan):.6e}")
    print(f"  gap_min: {parts.get('gap_min', np.nan):.6e}")
    print(f"  tip:     {np.asarray(result.tip)}")
    print(f"  ||u||:   {np.linalg.norm(result.u_flat_opt):.6e}")


def summarise_model_cache(model, *, prefix: str = "[MODEL CACHE]") -> None:
    """
    Print cache summary for a MagneticBeamForwardModel.
    """
    cache = model.cache

    print(prefix)
    print(f"  empty:    {cache.is_empty()}")
    print(f"  L_model:  {cache.L_model}")
    print(f"  p7_last:  {cache.p7_last}")
    print(f"  tip:      {cache.tip}")

    if cache.u_flat_opt is not None:
        print(f"  ||u||:    {np.linalg.norm(cache.u_flat_opt):.6e}")

    if cache.info is not None:
        print(f"  success:  {cache.info.get('success')}")
        print(f"  W:        {cache.info.get('W', np.nan):.6e}")