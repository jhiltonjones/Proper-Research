import numpy as np
import scipy.sparse as sp
import osqp


def solve_qp_osqp(
    H,
    f,
    A=None,
    l=None,
    u=None,
    U_warm=None,
    *,
    eps_abs=1e-5,
    eps_rel=1e-5,
    max_iter=10000,
    polish=True,
    verbose=False,
):
    """
    Solve a convex QP using OSQP.

    Problem form:

        minimize    0.5 z.T H z + f.T z
        subject to  l <= A z <= u

    Parameters
    ----------
    H:
        Quadratic cost matrix, shape (n, n)

    f:
        Linear cost vector, shape (n,)

    A:
        Constraint matrix, shape (n_constraints, n)

    l, u:
        Lower and upper constraint bounds.

    U_warm:
        Optional warm-start primal vector, shape (n,)

    Returns
    -------
    z_opt:
        Optimal primal vector, or None if solve failed.

    obj_val:
        Objective value, or np.inf if solve failed.

    status:
        OSQP status string.
    """
    H = np.asarray(H, float)
    f = np.asarray(f, float).reshape(-1)

    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(f"H must be square, got shape {H.shape}.")

    n = H.shape[0]

    if f.size != n:
        raise ValueError(f"f has length {f.size}, expected {n}.")

    # Symmetrise H for numerical safety.
    H = 0.5 * (H + H.T)

    if A is None:
        A = np.zeros((0, n), float)
        l = np.zeros(0, float)
        u = np.zeros(0, float)
    else:
        A = np.asarray(A, float)
        l = np.asarray(l, float).reshape(-1)
        u = np.asarray(u, float).reshape(-1)

        if A.ndim != 2:
            raise ValueError(f"A must be 2D, got shape {A.shape}.")

        if A.shape[1] != n:
            raise ValueError(
                f"A has {A.shape[1]} columns, expected {n}."
            )

        if A.shape[0] != l.size or A.shape[0] != u.size:
            raise ValueError(
                f"Constraint dimension mismatch: A={A.shape}, "
                f"l={l.shape}, u={u.shape}."
            )

    if np.any(l > u):
        bad = np.where(l > u)[0]
        raise ValueError(
            f"OSQP bounds invalid: l > u at rows {bad[:10]}."
        )

    P = sp.csc_matrix(H)
    q = f
    A_sp = sp.csc_matrix(A)

    solver = osqp.OSQP()
    solver.setup(
        P=P,
        q=q,
        A=A_sp,
        l=l,
        u=u,
        eps_abs=float(eps_abs),
        eps_rel=float(eps_rel),
        max_iter=int(max_iter),
        polish=bool(polish),
        verbose=bool(verbose),
    )

    if U_warm is not None:
        U_warm = np.asarray(U_warm, float).reshape(-1)

        if U_warm.size != n:
            raise ValueError(
                f"U_warm has length {U_warm.size}, expected {n}."
            )

        solver.warm_start(x=U_warm)

    result = solver.solve()
    status = str(result.info.status).lower()

    if result.x is None:
        return None, np.inf, status

    z_opt = np.asarray(result.x, float).reshape(-1)

    obj_val = float(0.5 * z_opt @ H @ z_opt + f @ z_opt)

    return z_opt, obj_val, status