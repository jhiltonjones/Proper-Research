from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np
import scipy.sparse as sp
import osqp


@dataclass(frozen=True)
class PersistentOSQPSettings:
    eps_abs: float = 1.0e-5
    eps_rel: float = 1.0e-5
    max_iter: int = 100000
    polishing: bool = False
    verbose: bool = False
    adaptive_rho: bool = True


class PersistentOSQPSolver:
    """
    Reuse one OSQP workspace while the QP dimensions remain unchanged.

    The matrices use a fixed dense sparsity pattern, including explicit zeros.
    This is deliberate: MPC coefficients can change from zero to non-zero between
    stages or SQP iterations, while OSQP matrix updates require an unchanged
    sparsity pattern.
    """

    def __init__(self, settings: PersistentOSQPSettings | None = None):
        self.settings = settings or PersistentOSQPSettings()
        self._solver: osqp.OSQP | None = None
        self._shape: tuple[int, int] | None = None
        self._n: int | None = None
        self._m: int | None = None
        self.setup_count = 0
        self.update_count = 0
        self.solve_count = 0
        self.setup_time_total_s = 0.0
        self.update_time_total_s = 0.0
        self.solve_time_total_s = 0.0
        self.last_x: np.ndarray | None = None
        self.last_y: np.ndarray | None = None

    def reset(self) -> None:
        self._solver = None
        self._shape = None
        self._n = None
        self._m = None
        self.last_x = None
        self.last_y = None

    @staticmethod
    def _validate(
        H,
        f,
        A,
        l,
        u,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        H = H.toarray() if sp.issparse(H) else np.asarray(H, dtype=float)
        f = np.asarray(f, dtype=float).reshape(-1)
        A = A.toarray() if sp.issparse(A) else np.asarray(A, dtype=float)
        l = np.asarray(l, dtype=float).reshape(-1)
        u = np.asarray(u, dtype=float).reshape(-1)

        if H.ndim != 2 or H.shape[0] != H.shape[1]:
            raise ValueError(f"H must be square, got {H.shape}.")
        n = H.shape[0]
        if f.size != n:
            raise ValueError(f"f has length {f.size}; expected {n}.")
        if A.ndim != 2 or A.shape[1] != n:
            raise ValueError(f"A must have shape (m, {n}); got {A.shape}.")
        if l.size != A.shape[0] or u.size != A.shape[0]:
            raise ValueError(
                f"Bounds must have length {A.shape[0]}; got l={l.size}, u={u.size}."
            )
        if np.any(l > u):
            raise ValueError("OSQP bounds contain l > u.")
        if not np.all(np.isfinite(H)) or not np.all(np.isfinite(f)):
            raise FloatingPointError("H or f contains non-finite values.")
        if not np.all(np.isfinite(A)):
            raise FloatingPointError("A contains non-finite values.")

        # OSQP consumes only the upper triangular part of P.
        H = 0.5 * (H + H.T)
        return H, f, A, l, u

    @staticmethod
    def _fixed_pattern_P(H: np.ndarray) -> sp.csc_matrix:
        n = H.shape[0]
        rows, cols = np.triu_indices(n)
        return sp.csc_matrix(
            (H[rows, cols], (rows, cols)),
            shape=(n, n),
        )

    @staticmethod
    def _fixed_pattern_A(A: np.ndarray) -> sp.csc_matrix:
        m, n = A.shape
        rows = np.repeat(np.arange(m, dtype=np.int32), n)
        cols = np.tile(np.arange(n, dtype=np.int32), m)
        return sp.csc_matrix(
            (A.reshape(-1), (rows, cols)),
            shape=(m, n),
        )

    def _setup(
        self,
        P: sp.csc_matrix,
        f: np.ndarray,
        A: sp.csc_matrix,
        l: np.ndarray,
        u: np.ndarray,
    ) -> None:
        solver = osqp.OSQP()
        settings_new: dict[str, Any] = {
            "eps_abs": float(self.settings.eps_abs),
            "eps_rel": float(self.settings.eps_rel),
            "max_iter": int(self.settings.max_iter),
            "polishing": bool(self.settings.polishing),
            "verbose": bool(self.settings.verbose),
            "adaptive_rho": bool(self.settings.adaptive_rho),
            "warm_starting": True,
        }
        started = time.perf_counter()
        try:
            solver.setup(P=P, q=f, A=A, l=l, u=u, **settings_new)
        except (TypeError, ValueError):
            # Compatibility with older OSQP Python releases.
            settings_old = dict(settings_new)
            settings_old["polish"] = settings_old.pop("polishing")
            settings_old["warm_start"] = settings_old.pop("warm_starting")
            solver.setup(P=P, q=f, A=A, l=l, u=u, **settings_old)
        self.setup_time_total_s += time.perf_counter() - started
        self.setup_count += 1
        self._solver = solver
        self._shape = (P.shape[0], A.shape[0])
        self._n = P.shape[0]
        self._m = A.shape[0]

    def solve(
        self,
        H,
        f,
        A,
        l,
        u,
        *,
        x_warm: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, float, str, dict[str, Any]]:
        H, f, A_dense, l, u = self._validate(H, f, A, l, u)
        P = self._fixed_pattern_P(H)
        A_csc = self._fixed_pattern_A(A_dense)

        required_shape = (P.shape[0], A_csc.shape[0])
        setup_performed = self._solver is None or self._shape != required_shape

        if setup_performed:
            self._setup(P, f, A_csc, l, u)
        else:
            assert self._solver is not None
            started = time.perf_counter()
            try:
                self._solver.update(
                    Px=P.data,
                    Ax=A_csc.data,
                    q=f,
                    l=l,
                    u=u,
                )
            except Exception:
                # Be conservative: rebuild if the installed OSQP rejects an
                # update because of version or pattern details.
                self.reset()
                self._setup(P, f, A_csc, l, u)
                setup_performed = True
            else:
                self.update_time_total_s += time.perf_counter() - started
                self.update_count += 1

        assert self._solver is not None

        if x_warm is not None:
            x0 = np.asarray(x_warm, dtype=float).reshape(-1)
            if x0.size != H.shape[0]:
                raise ValueError(
                    f"x_warm has length {x0.size}; expected {H.shape[0]}."
                )
            try:
                if self.last_y is not None and self.last_y.size == A_dense.shape[0]:
                    self._solver.warm_start(x=x0, y=self.last_y)
                else:
                    self._solver.warm_start(x=x0)
            except TypeError:
                self._solver.warm_start(x=x0)

        started = time.perf_counter()
        try:
            result = self._solver.solve(raise_error=False)
        except TypeError:
            result = self._solver.solve()
        wall_s = time.perf_counter() - started
        self.solve_time_total_s += wall_s
        self.solve_count += 1

        info = getattr(result, "info", None)
        status = str(getattr(info, "status", "unknown"))
        obj_val = float(getattr(info, "obj_val", np.inf))

        x = getattr(result, "x", None)
        y = getattr(result, "y", None)
        if x is not None:
            x = np.asarray(x, dtype=float).reshape(-1)
            if np.all(np.isfinite(x)):
                self.last_x = x.copy()
            else:
                x = None
        if y is not None:
            y_arr = np.asarray(y, dtype=float).reshape(-1)
            if np.all(np.isfinite(y_arr)):
                self.last_y = y_arr.copy()

        diagnostics = {
            "osqp_setup_performed": int(setup_performed),
            "osqp_workspace_reused": int(not setup_performed),
            "osqp_primal_dim": int(H.shape[0]),
            "osqp_constraint_dim": int(A_dense.shape[0]),
            "osqp_setup_count": int(self.setup_count),
            "osqp_update_count": int(self.update_count),
            "osqp_solve_count": int(self.solve_count),
            "osqp_wall_s": float(wall_s),
            "osqp_iter": int(getattr(info, "iter", -1)),
            "osqp_run_time_s": float(getattr(info, "run_time", np.nan)),
            "osqp_setup_time_s": float(getattr(info, "setup_time", np.nan)),
            "osqp_solve_time_s": float(getattr(info, "solve_time", np.nan)),
            "osqp_update_time_s": float(getattr(info, "update_time", np.nan)),
            "osqp_status": status,
        }
        return x, obj_val, status, diagnostics
