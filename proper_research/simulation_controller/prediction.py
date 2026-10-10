import numpy as np


def seq_mat_ltv(A, B_list):
    """
    Build stacked prediction matrices for the LTV system:

        x_{k+1} = A x_k + B_k u_k

    Returns:
        Mx: (N*n, n)
        Mc: (N*n, N*m)
    """
    B_list = [np.asarray(B, float) for B in B_list]
    N = len(B_list)

    if N == 0:
        raise ValueError("B_list must be non-empty.")

    A = np.asarray(A, float)
    n, m = B_list[0].shape

    if A.shape != (n, n):
        raise ValueError(f"A has shape {A.shape}, expected {(n, n)}.")

    for k, Bk in enumerate(B_list):
        if Bk.shape != (n, m):
            raise ValueError(
                f"B_list[{k}] has shape {Bk.shape}, expected {(n, m)}."
            )

    Mx = np.zeros((N * n, n), float)
    Mc = np.zeros((N * n, N * m), float)

    # A_powers[q] = A^q
    A_powers = [np.eye(n)]
    for _ in range(N):
        A_powers.append(A @ A_powers[-1])

    for i in range(N):
        row = slice(i * n, (i + 1) * n)

        # x_{i+1} = A^{i+1} x0 + sum_{j=0}^{i} A^{i-j} B_j u_j
        Mx[row, :] = A_powers[i + 1]

        for j in range(i + 1):
            col = slice(j * m, (j + 1) * m)
            Mc[row, col] = A_powers[i - j] @ B_list[j]

    return Mx, Mc


def seq_mat_lti(A, B, N):
    """
    Build stacked prediction matrices for the LTI system:

        x_{k+1} = A x_k + B u_k

    Returns:
        Mx: (N*n, n)
        Mc: (N*n, N*m)
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)

    if N <= 0:
        raise ValueError("N must be positive.")

    n, m = B.shape

    if A.shape != (n, n):
        raise ValueError(f"A has shape {A.shape}, expected {(n, n)}.")

    Mx = np.zeros((N * n, n), float)
    Mc = np.zeros((N * n, N * m), float)

    A_powers = [np.eye(n)]
    for _ in range(N):
        A_powers.append(A @ A_powers[-1])

    for i in range(N):
        row = slice(i * n, (i + 1) * n)

        Mx[row, :] = A_powers[i + 1]

        for j in range(i + 1):
            col = slice(j * m, (j + 1) * m)
            Mc[row, col] = A_powers[i - j] @ B

    return Mx, Mc


def build_Du_matrix(Np: int, m: int):
    """
    Build first-difference matrix for control smoothing.

    For U = [u0, u1, ..., u_{Np-1}], this returns Du such that:

        Du @ U = [u1-u0, u2-u1, ..., u_{Np-1}-u_{Np-2}]
    """
    Np = int(Np)
    m = int(m)

    if Np <= 0:
        raise ValueError("Np must be positive.")

    if m <= 0:
        raise ValueError("m must be positive.")

    if Np <= 1:
        return np.zeros((0, Np * m), float)

    D1 = np.zeros((Np - 1, Np), float)

    for i in range(Np - 1):
        D1[i, i] = -1.0
        D1[i, i + 1] = 1.0

    return np.kron(D1, np.eye(m))