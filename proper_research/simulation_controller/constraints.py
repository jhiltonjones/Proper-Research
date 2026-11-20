import numpy as np


def cumulative_state_control_matrix(
    *,
    Np,
    n_state,
    n_control,
    dt,
    state_rate_matrix=None,
):
    """Return the stacked discrete integrator from controls to states.

    For

        s[k + 1] = s[k] + dt * E @ u[k],

    this returns ``S`` such that

        [s[1], ..., s[Np]] = tile(s[0], Np) + S @ U.

    ``state_rate_matrix`` is ``E``.  The joint-plus-insertion controller uses
    ``E = I(7)`` with

        s = [q1, ..., q6, insertion]
        u = [qd1, ..., qd6, insertion_rate].
    """
    Np = int(Np)
    n_state = int(n_state)
    n_control = int(n_control)
    dt = float(dt)

    if Np <= 0:
        raise ValueError("Np must be positive.")
    if n_state <= 0 or n_control <= 0:
        raise ValueError("n_state and n_control must be positive.")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive.")

    if state_rate_matrix is None:
        if n_state != n_control:
            raise ValueError(
                "state_rate_matrix is required when n_state != n_control."
            )
        E = np.eye(n_state)
    else:
        E = np.asarray(state_rate_matrix, float)
        if E.shape != (n_state, n_control):
            raise ValueError(
                "state_rate_matrix must have shape "
                f"{(n_state, n_control)}, got {E.shape}."
            )
        if not np.all(np.isfinite(E)):
            raise ValueError("state_rate_matrix contains non-finite values.")

    cumulative = np.tril(np.ones((Np, Np), dtype=float))
    return dt * np.kron(cumulative, E)


def integrated_state_bounds(
    *,
    state0,
    state_min,
    state_max,
    Np,
    dt,
    n_control=None,
    state_rate_matrix=None,
):
    """Build horizon-wide bounds for an integrated state.

    The returned OSQP block enforces

        state_min <= state[k] <= state_max,  k = 1, ..., Np

    for the discrete integrator used by ``cumulative_state_control_matrix``.
    Infinite entries are supported, so individual state channels may be left
    unconstrained.
    """
    state0 = np.asarray(state0, float).reshape(-1)
    state_min = np.asarray(state_min, float).reshape(-1)
    state_max = np.asarray(state_max, float).reshape(-1)

    if state_min.shape != state0.shape or state_max.shape != state0.shape:
        raise ValueError(
            "state0, state_min and state_max must have identical shapes."
        )
    if not np.all(np.isfinite(state0)):
        raise ValueError("state0 contains non-finite values.")
    if np.any(np.isnan(state_min)) or np.any(np.isnan(state_max)):
        raise ValueError("state bounds may be infinite but not NaN.")
    if np.any(state_min > state_max):
        raise ValueError("Some state lower bounds exceed upper bounds.")

    n_state = state0.size
    if n_control is None:
        n_control = n_state

    A = cumulative_state_control_matrix(
        Np=Np,
        n_state=n_state,
        n_control=int(n_control),
        dt=dt,
        state_rate_matrix=state_rate_matrix,
    )
    state0_stack = np.tile(state0, int(Np))
    lower = np.tile(state_min, int(Np)) - state0_stack
    upper = np.tile(state_max, int(Np)) - state0_stack
    return A, lower, upper


def control_rate_bounds(*, u_previous, rate_max, Np, dt):
    """Build hard bounds on changes of a velocity-like MPC control.

    ``rate_max`` has acceleration units.  The constraints are

        |u[0] - u_previous| <= rate_max * dt
        |u[k] - u[k - 1]|   <= rate_max * dt.

    For the joint-plus-insertion controller, the first six entries constrain
    joint acceleration and the seventh constrains insertion acceleration.
    Infinite entries disable selected channels.
    """
    u_previous = np.asarray(u_previous, float).reshape(-1)
    rate_max = np.asarray(rate_max, float).reshape(-1)
    Np = int(Np)
    dt = float(dt)

    if u_previous.shape != rate_max.shape:
        raise ValueError("u_previous and rate_max must have identical shapes.")
    if Np <= 0:
        raise ValueError("Np must be positive.")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive.")
    if not np.all(np.isfinite(u_previous)):
        raise ValueError("u_previous contains non-finite values.")
    if np.any(np.isnan(rate_max)) or np.any(rate_max < 0.0):
        raise ValueError("rate_max entries must be non-negative or +inf.")

    m = u_previous.size
    Nu = Np * m
    A = np.zeros((Nu, Nu), dtype=float)
    for stage in range(Np):
        rows = slice(stage * m, (stage + 1) * m)
        cols = slice(stage * m, (stage + 1) * m)
        A[rows, cols] = np.eye(m)
        if stage > 0:
            previous = slice((stage - 1) * m, stage * m)
            A[rows, previous] = -np.eye(m)

    centre = np.zeros(Nu, dtype=float)
    centre[:m] = u_previous
    maximum_change = np.tile(rate_max * dt, Np)
    return A, centre - maximum_change, centre + maximum_change


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
    if np.any(np.isnan(u_max)) or np.any(u_max < 0.0):
        raise ValueError("u_max entries must be non-negative or +inf.")

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
