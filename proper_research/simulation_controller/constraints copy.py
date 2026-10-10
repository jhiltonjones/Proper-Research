import numpy as np


def trust_region_bounds(U_guess, trust_radius):
    """
    Bounds for the SQP trust region:

        U_guess - trust_radius <= U <= U_guess + trust_radius

    U_guess:
        shape (Np, m)

    trust_radius:
        None, scalar, shape (m,), or shape (Np*m,)
    """
    U_guess = np.asarray(U_guess, float)

    if U_guess.ndim != 2:
        raise ValueError(
            f"U_guess must have shape (Np, m), got {U_guess.shape}."
        )

    Np, m = U_guess.shape

    if trust_radius is None:
        lower = -np.full(Np * m, np.inf)
        upper = +np.full(Np * m, np.inf)
        return lower, upper

    tr = np.asarray(trust_radius, float).reshape(-1)

    if tr.size == 1:
        tr_stage = np.full((Np, m), float(tr[0]))
    elif tr.size == m:
        tr_stage = np.tile(tr.reshape(1, m), (Np, 1))
    elif tr.size == Np * m:
        tr_stage = tr.reshape(Np, m)
    else:
        raise ValueError(
            "trust_radius must be None, scalar, length m, or length Np*m."
        )

    if np.any(tr_stage < 0.0):
        raise ValueError("trust_radius entries must be non-negative.")

    lower = (U_guess - tr_stage).reshape(-1)
    upper = (U_guess + tr_stage).reshape(-1)

    return lower, upper


def input_bounds(u_max, Np):
    """
    Build box bounds:

        -u_max <= U <= u_max

    Returns:
        A, l, u
    """
    u_max = np.asarray(u_max, float).reshape(-1)
    Np = int(Np)
    m = u_max.size

    if Np <= 0:
        raise ValueError("Np must be positive.")

    Nu = Np * m

    A = np.eye(Nu)
    bound = np.tile(u_max, Np)

    l = -bound
    u = +bound

    return A, l, u


def dL_bounds(Np, m, dL_index, dL_back_max, dL_fwd_max=np.inf):
    """
    Build insertion-rate bounds for dL control:

        -dL_back_max <= U[k, dL_index] <= dL_fwd_max

    Returns:
        A, l, u
    """
    Np = int(Np)
    m = int(m)
    dL_index = int(dL_index)

    if Np <= 0:
        raise ValueError("Np must be positive.")

    if m <= 0:
        raise ValueError("m must be positive.")

    if not 0 <= dL_index < m:
        raise ValueError(f"dL_index={dL_index} outside control dimension m={m}.")

    Nu = Np * m

    A = np.zeros((Np, Nu), float)

    for k in range(Np):
        A[k, k * m + dL_index] = 1.0

    l = -float(dL_back_max) * np.ones(Np)
    u = +float(dL_fwd_max) * np.ones(Np)

    return A, l, u


def box_constraint_from_bounds(lower, upper):
    """
    Convert lower <= U <= upper into OSQP form:

        A U between l and u

    Returns:
        A, l, u
    """
    lower = np.asarray(lower, float).reshape(-1)
    upper = np.asarray(upper, float).reshape(-1)

    if lower.shape != upper.shape:
        raise ValueError(
            f"lower and upper shape mismatch: {lower.shape} vs {upper.shape}."
        )

    if np.any(lower > upper):
        raise ValueError("Some lower bounds are greater than upper bounds.")

    A = np.eye(lower.size)

    return A, lower, upper


def stack_constraints(blocks):
    """
    Stack a list of OSQP constraint blocks.

    Each block is:
        (A, l, u)

    Returns:
        A, l, u
    """
    blocks = [b for b in blocks if b is not None]

    if len(blocks) == 0:
        return (
            np.zeros((0, 0), float),
            np.zeros(0, float),
            np.zeros(0, float),
        )

    A_list, l_list, u_list = [], [], []

    n_cols = None

    for A, l, u in blocks:
        A = np.asarray(A, float)
        l = np.asarray(l, float).reshape(-1)
        u = np.asarray(u, float).reshape(-1)

        if A.ndim != 2:
            raise ValueError("A must be two-dimensional.")

        if A.shape[0] != l.size or A.shape[0] != u.size:
            raise ValueError(
                f"Constraint shape mismatch: A={A.shape}, l={l.shape}, u={u.shape}."
            )

        if n_cols is None:
            n_cols = A.shape[1]
        elif A.shape[1] != n_cols:
            raise ValueError(
                f"Constraint column mismatch: got {A.shape[1]}, expected {n_cols}."
            )

        A_list.append(A)
        l_list.append(l)
        u_list.append(u)

    return (
        np.vstack(A_list),
        np.concatenate(l_list),
        np.concatenate(u_list),
    )